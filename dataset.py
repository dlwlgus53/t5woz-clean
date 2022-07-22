import re
import pdb
import json
import torch
import pickle
import ontology
from tqdm import tqdm
import logging
from log_conf import init_logger
from collections import defaultdict
import random
logger = logging.getLogger("my")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, data_type):
        random.seed(args.seed)
        self.data_type = data_type
        self.tokenizer = args.tokenizer
        self.max_length = args.max_length
        self.data_type = data_type
        
        
        if self.data_type == 'train':
            raw_path = f'{data_path[:-5]}{args.data_rate}.json'
        else:
            raw_path = f'{data_path}'
        
        if args.do_short:
            raw_path = f'../KLUE/train_data_short.json' 
            
            
        logger.info(f"load {self.data_type} raw file {raw_path}")   
        raw_dataset = json.load(open(raw_path , "r"))
        turn_id, dial_id,  question, schema, answer, gold_belief_state, gold_context= self.seperate_data(raw_dataset)

        assert len(turn_id) == len(dial_id) == len(question)\
            == len(schema) == len(answer)
            
        self.answer = answer # for debugging
        self.target = self.encode(answer)
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.schema = schema
        self.gold_belief_state = gold_belief_state
        self.gold_context = gold_context
        
            
            
            
    def encode(self, texts ,return_tensors="pt"):
        examples = []
        for i, text in enumerate(texts):
            # Truncate
            while True:
                tokenized = self.tokenizer.batch_encode_plus([text], padding=False, return_tensors=return_tensors) # TODO : special token
                if len(tokenized)> self.max_length:
                    idx = [m.start() for m in re.finditer("\[user\]", text)]
                    text = text[:idx[0]] + text[idx[1]:] # delete one turn
                else:
                    break
                
            examples.append(tokenized)
        return examples

    def __len__(self):
        return len(self.dial_id)

    def seperate_data(self, dataset):
        gold_belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id
        gold_context= defaultdict(lambda : defaultdict(str))# dial_id, # turn_id
        
        question = []
        answer = []
        schema = []
        dial_id = []
        turn_id = []
        
        for d_id in dataset.keys():
            dialogue = dataset[d_id]
            dialogue_text = ""
            for t_id, turn in enumerate(dialogue):
                dialogue_text += '[사용자] '
                dialogue_text += turn['user']
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    q = ontology.QA[key]['description']

                    if key in turn['belief']: # 언급을 한 경우
                        a = turn['belief'][key]
                        if isinstance(a, list) : a= a[0] # in muptiple type, a == ['sunday',6]
                    else:a = ontology.QA['NOT_MENTIONED']
                    
                    schema.append(key)
                    answer.append(a)
                    question.append(q)
                    dial_id.append(d_id)
                    turn_id.append(t_id)
                gold_context[d_id][t_id] = dialogue_text

                dialogue_text += '[시스템] '
                dialogue_text += turn['system']
                
                    
        for_sort = [[t,d,q,s,a] for (t,d,q,s,a) in zip(turn_id, dial_id,  question, schema, answer)]
        sorted_items = sorted(for_sort, key=lambda x: (x[0], x[1]))
        
        turn_id = [s[0] for s in sorted_items]
        dial_id = [s[1] for s in sorted_items]
        question = [s[2] for s in sorted_items]
        schema_sort = [s[3] for s in sorted_items]
        answer = [s[4] for s in sorted_items]
        
        
        # sort guaranteed to be stable : it is important because of question!   
        # assert schema_sort == schema
        return turn_id, dial_id,  question, schema, answer, gold_belief_state, gold_context

    def __getitem__(self, index):
        dial_id = self.dial_id[index]
        turn_id = self.turn_id[index]
        schema = self.schema[index]
        question = self.question[index]
        gold_context = self.gold_context[index]

        
        target = {k:v.squeeze() for (k,v) in self.target[index].items()}
        
        return {"target": target,"turn_id" : turn_id,"question" : question, "gold_context" : gold_context,\
            "dial_id" : dial_id, "schema":schema }
    


    
    def make_DB(self, belief_state, activate):
        pass
    
    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        """

        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        question = [x["question"] for x in batch]
        schema = [x["schema"] for x in batch]
        target_list = [x["target"] for x in batch]
        

        history = [self.gold_context[d][t] for (d,t) in zip(dial_id, turn_id)]
        input_source = [f"question: {q} context: {c}" for (q,c) in  \
            zip(question, history, )]
        
        source = self.encode(input_source)
        source_list = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
            
        pad_source = self.tokenizer.pad(source_list,padding=True)
        pad_target = self.tokenizer.pad(target_list,padding=True)
        
        return {"input": pad_source, "target": pad_target,\
                 "schema":schema, "dial_id":dial_id, "turn_id":turn_id}
        

if __name__ == '__main__':
    import argparse
    init_logger(f'data_process.log')
    logger = logging.getLogger("my")

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_rate' ,  type = float, default=1.0)
    parser.add_argument('--do_short' ,  type = int, default=1)
    parser.add_argument('--dst_student_rate' ,  type = float, default=0.0)
    parser.add_argument('--seed' ,  type = float, default=1)
    parser.add_argument('--aux' ,  type = int, default=1)
    
    parser.add_argument('--max_length' ,  type = int, default=128)
    
    
    args = parser.parse_args()

    args.data_path = '../KLUE/dev_data_short.json'
    from transformers import T5Tokenizer
    args.tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
    
    dataset = Dataset(args, args.data_path, 'train')
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    t = args.tokenizer
    for batch in loader:
        for i in range(16):
            print(t.decode(batch['input']['input_ids'][i]))
            print(t.decode(batch['target']['input_ids'][i]))
            print()
            
        pdb.set_trace()
    
    