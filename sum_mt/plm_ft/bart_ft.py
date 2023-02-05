import math
import os
from args import pretrain_args
import nltk
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    DataCollatorForSeq2Seq,
    get_scheduler,
    set_seed,
)
import time, sys
from nltk import word_tokenize

from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer
import logging
logger = logging.getLogger()
logFormatter = logging.Formatter("%(asctime)s [%(funcName)-15s] [%(levelname)-5.5s]  %(message)s")
consoleHandler = logging.StreamHandler(); consoleHandler.setFormatter(logFormatter); logger.addHandler(consoleHandler); logger.setLevel('INFO');

sys.path.append(str(Path(__file__).absolute().parent.parent))
import htx_utils

class BART:
    def __init__(self, args, base = 'facebook/bart-large', checkpoint = 'facebook/bart-large'):
        self.args = args
        if base.startswith('facebook/bart'):
            mC, tC =  BartForConditionalGeneration, BartTokenizer; args.model_base_simple = 'bart';
        if base.startswith('t5'):
            mC, tC =  T5ForConditionalGeneration, T5Tokenizer; args.model_base_simple = 't5';

        logger.info('initializing model from %s', checkpoint)
        self.model = mC.from_pretrained(checkpoint, local_files_only = True)
        logger.info('initializing tokenizer from %s', base)
        self.tokenizer = tC.from_pretrained(base, local_files_only = True)
        self.criterion = None

        # Initialize the accelerator. We will let the accelerator handle device placement for us
        # in this example
        accelerator = Accelerator(split_batches=True); self.accelerator = accelerator;

        data_files = {
            'train': args.train_file,
            'validation': args.validation_file
        }
        extension = args.train_file.split('.')[-1]
        raw_datasets = load_dataset(extension, data_files=data_files); self.raw_datasets = raw_datasets;

        # Preprocessing the datasets
        # First we tokenize all the texts
        column_names = raw_datasets['train'].column_names
        text_column, summary_column = column_names[0], column_names[1]

        # Temporarily set max_target_length for training
        padding = False

        def preprocess_function(examples):
            inputs = examples[text_column]
            targets = examples[summary_column]
            inputs = [inp for inp in inputs]
            #do tokenization for each sample, with no padding
            model_inputs = self.tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=args.max_target_length, padding=padding, truncation=True)
            
            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        load_fn = f'{args.dataset}_saves/processed_datasets_{args.model_base_simple}.save'
        logger.info('loading processed_datasets from %s', load_fn)
        processed_datasets = torch.load(load_fn) 
        #processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=column_names, load_from_cache_file=True)

        train_dataset = processed_datasets["train"]; eval_dataset = processed_datasets["validation"];
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        
        #args.ignore_pad_token_for_loss is True
        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None, #use_fp16 is False
        )

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        print(f'using train_batch_size {args.per_device_train_batch_size} also for eval_dataloader')
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Prepare everything with our `accelerator`. # this step converts model to cuda
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader
        )

    def evaluate(self, model, eval_dataloader, do_report = False, max_step = -1): 
        #note that the model in pretrain is wrapped by accelerate, it's not the original model
        model.eval()
        loss_lis = []
        for step, batch in enumerate(eval_dataloader):      
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                loss_lis.append(loss.detach().item())
            if step > max_step: break
        torch.cuda.empty_cache()

        res = {'loss': np.mean(loss_lis)}
        if do_report:
            logger.info('evaluate loss: %f', res['loss'])
        return res
            
    def pretrain(self, args):
        """
        args.seed
        args.datapath
        args.max_source_length
        args.max_target_length
        args.ignore_pad_token_for_loss
        """

        set_seed(args.seed)

        model, optimizer, train_dataloader, eval_dataloader, accelerator = self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.accelerator
        train_dataset, eval_dataset = self.train_dataset, self.eval_dataset

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
        
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        s_d = {'loss': [], 'lr': [],} 
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(args.num_train_epochs):
            logger.info('begin epoch %d', epoch)
            for step, batch in enumerate(train_dataloader):
                model.train()
                #the input/ouptut batch uses right-padding (better for positional embedding?)
                outputs = model(**batch)
                loss = outputs.loss
                s_d['loss'].append(loss.item()); s_d['lr'] = [lr_scheduler.get_last_lr()[0]]; #log the loss before it got divided
                loss = loss / args.gradient_accumulation_steps #i think this looks right, make it the same as using a larger bz 
                accelerator.backward(loss)
                
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    if completed_steps % 200 == 0 or (completed_steps <= 100 and completed_steps % 20 == 0):
                        log_str = 'step:' + str(completed_steps) + ' '
                        for tt in s_d:
                            log_str += tt + ': ' + str(np.mean(s_d[tt])) + ' '; s_d[tt] = [];
                        logger.info(log_str)

                    if args.save_every > 0:
                        if completed_steps % args.save_every == 0:
                            out_dir = f'{args.output_dir}/{completed_steps}'
                            logger.info('saving to %s', out_dir)
                            os.makedirs(out_dir, exist_ok=True)
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(out_dir, save_function=accelerator.save)
                    
                    if completed_steps % 1000 == 0 or (completed_steps <= 500 and completed_steps % 100 == 0) or (step == len(train_dataloader) - 1):
                        logger.info('doing evaluate at step %d', completed_steps)
                        self.evaluate(model, eval_dataloader, max_step = 2000, do_report = True)

                if completed_steps >= args.max_train_steps:
                    break

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            logger.info('saving to %s', args.output_dir)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

    def decode(self, ld_lines, max_num = None):
        self.model.eval(); args = self.args; tokenizer = self.tokenizer;
        #for debug #src_lines = [ww['text'] for ww in self.raw_datasets['train']]; ref_lines = [ww['summary'] for ww in self.raw_datasets['train']];
        dec_res = []; src_lines, ref_lines = ld_lines['src_lines'], ld_lines['ref_lines'];

        logger.info('decoding for %d src_lines', len(src_lines))
        for idx, line in enumerate(src_lines):
            if max_num is not None and idx >= max_num:
                break
            src_tokens = self.tokenizer(line, max_length=args.max_source_length, padding=False, truncation=True)['input_ids']
            summary_ids = self.model.generate(torch.LongTensor(src_tokens).unsqueeze(0).cuda(), do_sample = False, num_beams=5, min_length=0, max_length = 150) # if i don't set this, it will be default to 20 
            dec_line = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            dec_res.append({'idx': idx, 'src_line': line, 'dec_line': dec_line, 'ref_line': ref_lines[idx]})
            if idx <= 3:
                logger.info('decode idx %d dec: %s', idx, dec_line)
            if idx % 100 == 0:
                logger.info('decode idx %d')
        logger.info('%d line decoded.', len(dec_res))

        rouge_res = htx_utils.ref_rouge([ww['ref_line'] for ww in dec_res], [ww['dec_line'] for ww in dec_res])  
        logger.info('rouge_score: %s', str(rouge_res))

        return dec_res

def compute_freq(args):
    data_files = {
        'train': args.train_file,
        'validation': args.validation_file,
        'test': args.test_file,
    }
    extension = args.train_file.split('.')[-1]
    raw_datasets = load_dataset(extension, data_files=data_files);
    
    train_summarys = [s['summary'] for s in raw_datasets['train']]
    logger.info('saving train_summaries to cnndm_saves/train_summaries.save')
    torch.save(train_summarys, 'cnndm_saves/train_summaries.save')

    w_d = {}
    for split in raw_datasets:
        logger.info('computing freq with split %s', split)
        for sample in raw_datasets[split]:
            for w in word_tokenize(sample['text']):
                w = w.lower()
                if not w in w_d: w_d[w] = 0
                w_d[w] = w_d[w] + 1
    
    logger.info('%d tokens found, printing first 10...', len(w_d))
    for w, v in sorted(w_d.items(), key = lambda x: x[1], reverse = True)[:10]:
        logger.info('%s : %d', w, v)
    
    save_fn = 'cnndm_saves/wfreq.save'
    logger.info('saving wfreq to %s', save_fn)
    torch.save(w_d, save_fn)

if __name__ == '__main__':
    args = pretrain_args()
    logger.info('command is %s', args.command)

    if args.decode_src_load is None:
        args.decode_src_load = './saves/summeval_src_lines.save'

    if args.dataset == 'cnndm':
        if args.train_file is None: args.train_file = 'cnndm_saves/train.json';
        if args.validation_file is None: args.validation_file = 'cnndm_saves/validation.json';
        if args.test_file is None: args.test_file = 'cnndm_saves/test.json';

    if args.output_dir is None and args.command == 'train':
        mn = args.model_base; 
        if '/' in mn: mn = mn.split('/')[-1]
        args.output_dir = f'./checkpoints/{args.dataset}/{mn}/ep{args.num_train_epochs}lr{str(args.learning_rate)}ba{args.per_device_eval_batch_size}gacc{args.gradient_accumulation_steps}'
        if args.debug: args.output_dir += '_debug'
        logger.info('output_dir set to %s', args.output_dir)

    if args.command == 'freq':
        logger.info('COMMAND FREQ')
        compute_freq(args)

    if args.command == 'train':
        logger.info('COMMAND TRAIN')
        bart = BART(args, base = args.model_base, checkpoint = args.model_base)
        bart.pretrain(args)

    if args.command == 'eval':
        logger.info('COMMAND EVAL')
        bart = BART(args, base = args.model_base, checkpoint = args.load_checkpoint)
        logger.info('begin evaluate')
        bart.evaluate(bart.model, bart.eval_dataloader, do_report = True, max_step = 2000)

    if args.command == 'decode':
        logger.info('COMMAND DECODE')
        bart = BART(args, base = args.model_base, checkpoint = args.load_checkpoint)
        if args.decode_src_load is not None:
            logger.info('loading src_lines from %s', args.decode_src_load)
            ld_lines = torch.load(args.decode_src_load)
        else:
            logger.info('using validation data as src_lines')
            ld_lines = {'src_lines': [ww['text'] for ww in bart.raw_datasets['validation']], 'ref_lines': [ww['summary'] for ww in bart.raw_datasets['validation']]}

        dec_res = bart.decode(ld_lines, max_num = 500)
        if args.decode_save is not None:
            logger.info('saving to %s', args.decode_save)
            torch.save(dec_res, args.decode_save)
                 
        
