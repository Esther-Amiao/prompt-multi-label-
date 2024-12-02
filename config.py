CONFIG = {
    # model
    # 'train_data_file': './data/train_n.csv',
    'cuda': 'cuda:1', 
    
    'lang': 'go',
    'model_name_or_path': './microsoft/graphcodebert-base1',
    'config_name': './microsoft/graphcodebert-base1',
    'tokenizer_name': './microsoft/graphcodebert-base1',

    'epoch': 6,
#     训练
#     'do_train': True,
#     'do_eval': False,   
#     测试
    'do_train': False,
    'do_eval': True,
    
#     prompt设计实验 
#     'train_model_path': 'Train_Model/wiki1-ep8/epoch_',
#     'train_model_path': 'Train_Model/wiki/epoch_',
#     'dev_model_path': 'Train_Model/add-useques-asplm1/epoch_6',
    
# #     外部知识引入的，是wiki2的训练集和wiki1的测试集
#     # train1_set_wikiprefix 是prompt后面直接加了外部知识
    
#     'train_data_file': '../data_codereview/data/GraphCodeBert_data/train1_set_wiki1.csv',
#     'dev_data_file': '../data_codereview/data/GraphCodeBert_data/test_set.csv',
    
    
    #     二分类
    'train_model_path': 'Train_Model/wuguanci_01/epoch_',
    'dev_model_path': 'Train_Model/question01_prompt_des/epoch_1',
    
    'train_data_file': '../data_codereview/data/GraphCodeBert_data/question_train1_set_01_wuguanci.csv',
    'dev_data_file': '../data_codereview/data/GraphCodeBert_data/question_dev1_set_01_des.csv',
    
    
#     mask位置, question任务的时候要去model文件里改mask相关参数
    'start_index': 5,
    'end_index': 8,
    'start_question_index': 15,
    'end_question_index': 16,
    
    'nl_length': 330,
    'code_length': 150,
    # 在text多少后进行拼接
    'maxlen_pre_concat': 250,
    'maxlen_text_concat': 250,
    'maxlen_code_concat': 300,
    'data_flow_length': 64,

    'train_batch_size': 4,
    'eval_batch_size': 4,
    'learning_rate': 1e-5,
    'max_grad_norm': 1.0,
    'num_train_epochs': 5,
    'seed': 123456,
}

# parser.add_argument("--do_train", action='store_true',
#                     help="Whether to run training.")
# parser.add_argument("--do_eval", action='store_true',
#                     help="Whether to run eval on the dev set.")
# parser.add_argument("--do_test", action='store_true',
#                     help="Whether to run eval on the test set.")
#
# # 32
# parser.add_argument("--train_batch_size", default=8, type=int,
#                     help="Batch size for training.")
# parser.add_argument("--eval_batch_size", default=4, type=int,
#                     help="Batch size for evaluation.")
# parser.add_argument("--learning_rate", default=1e-5, type=float,
#                     help="The initial learning rate for Adam.")
# parser.add_argument("--max_grad_norm", default=1.0, type=float,
#                     help="Max gradient norm.")
# parser.add_argument("--num_train_epochs", default=1, type=int,
#                     help="Total number of training epochs to perform.")
# parser.add_argument('--seed', type=int, default=123456,
#                     help="random seed for initialization")
