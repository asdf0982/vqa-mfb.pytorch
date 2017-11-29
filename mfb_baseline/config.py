import argparse
# vqa tools - get from https://github.com/VT-vision-lab/VQA
VQA_TOOLS_PATH = '/home/yuz/data/VQA/PythonHelperTools'
VQA_EVAL_TOOLS_PATH = '/home/yuz/data/VQA/PythonEvaluationTools'

# location of the data
VQA_PREFIX = '/home/yuz/data/VQA'

feat = 'pool5'
DATA_PATHS = {
    'train': {
        'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_train2014_questions.json',
        'ans_file': VQA_PREFIX + '/Annotations/mscoco_train2014_annotations.json',
        'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/train2014/COCO_train2014_'%feat
    },
    'val': {
        'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_val2014_questions.json',
        'ans_file': VQA_PREFIX + '/Annotations/mscoco_val2014_annotations.json',
        'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/val2014/COCO_val2014_'%feat
    },
    'test-dev': {
        'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test-dev2015_questions.json',
        'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/test2015/COCO_test2015_'%feat
    },
    'test': {
        'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test2015_questions.json',
        'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/test2015/COCO_test2015_'%feat
    },
    'genome': {
        'genome_file': VQA_PREFIX + '/Questions/OpenEnded_genome_train_questions.json',
        'features_prefix': VQA_PREFIX + '/Features/genome/feat_resnet-152/resnet_%s_bgrms_large/'%feat
    }
}
def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--TRAIN_GPU_ID', type=int, default=3)
    parser.add_argument('--TEST_GPU_ID', type=int, default=3)
    parser.add_argument('--SEED', type=int, default=-1)
    parser.add_argument('--BATCH_SIZE', type=int, default=200)
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=1000)
    parser.add_argument('--NUM_OUTPUT_UNITS', type=int, default=3000)
    parser.add_argument('--MAX_WORDS_IN_QUESTION', type=int, default=15)
    parser.add_argument('--MAX_ITERATIONS', type=int, default=50000)
    parser.add_argument('--PRINT_INTERVAL', type=int, default=100)
    parser.add_argument('--CHECKPOINT_INTERVAL', type=int, default=5000)
    parser.add_argument('--TESTDEV_INTERVAL', type=int, default=45000)
    parser.add_argument('--RESUME', type=bool, default=False)
    parser.add_argument('--RESUME_PATH', type=str, default='./data/***.pth')
    parser.add_argument('--VAL_INTERVAL', type=int, default=5000)
    parser.add_argument('--IMAGE_CHANNEL', type=int, default=2048)
    parser.add_argument('--INIT_LERARNING_RATE', type=float, default=0.0007)
    parser.add_argument('--DECAY_STEPS', type=int, default=20000)
    parser.add_argument('--DECAY_RATE', type=float, default=0.5)
    parser.add_argument('--MFB_FACTOR_NUM', type=int, default=5)
    parser.add_argument('--MFB_OUT_DIM', type=int, default=1000)
    parser.add_argument('--LSTM_UNIT_NUM', type=int, default=1024)
    parser.add_argument('--LSTM_DROPOUT_RATIO', type=float, default=0.3)
    parser.add_argument('--MFB_DROPOUT_RATIO', type=float, default=0.1)
    parser.add_argument('--TRAIN_DATA_SPLITS', type=str, default='train')
    parser.add_argument('--QUESTION_VOCAB_SPACE', type=str, default='train')
    parser.add_argument('--ANSWER_VOCAB_SPACE', type=str, default='train')
    args = parser.parse_args()
    return args