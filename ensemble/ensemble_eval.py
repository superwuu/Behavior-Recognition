import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import os
def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble') 
    parser.add_argument(
        '--mixformer_J_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument(
        '--mixformer_B_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument(
        '--mixformer_JM_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument(
        '--mixformer_BM_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument(
        '--mixformer_k2_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--mixformer_k2M_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--tegcn_J_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--tegcn_B_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--ctrgcn_J2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument(
        '--ctrgcn_B2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument(
        '--ctrgcn_JM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument(
        '--ctrgcn_BM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument(
        '--ctrgcn_J3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--ctrgcn_B3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--ctrgcn_JM3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--ctrgcn_BM3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--tdgcn_J2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument(
        '--tdgcn_B2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument(
        '--tdgcn_JM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument(
        '--tdgcn_BM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument(
        '--mstgcn_J2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument(
        '--mstgcn_B2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument(
        '--mstgcn_JM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument(
        '--mstgcn_BM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument(
        '--val_sample', 
        type = str,
        default = './Process_data/CS_test_V1.txt')
    parser.add_argument(
        '--benchmark', 
        type = str,
        default = 'V1')
    return parser

def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        fr = open(file,'rb') 
        inf = pickle.load(fr)

        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = torch.tensor(data = df.values)
        final_score += Rate[idx] * score
    return final_score

def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
            
    wrong_num = np.array(wrong_index).shape[0]
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label(val_txt_path):
    true_label = np.load(val_txt_path)

    true_label = torch.from_numpy(np.array(true_label))
    return true_label

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Mix_GCN Score File
    j_file = args.mixformer_J_Score
    b_file = args.mixformer_B_Score
    jm_file = args.mixformer_JM_Score
    bm_file = args.mixformer_BM_Score
    te_file = args.tegcn_J_Score
    teb_file = args.tegcn_B_Score
    k2_file = args.mixformer_k2_Score
    k2m_file = args.mixformer_k2M_Score

    j_file2 = args.ctrgcn_J2d_Score
    b_file2 = args.ctrgcn_B2d_Score

    jm3d_file = args.ctrgcn_JM3d_Score
    bm3d_file = args.ctrgcn_BM3d_Score
    
    j_file3 = args.tdgcn_J2d_Score
    b_file3 = args.tdgcn_B2d_Score
    jm_file3 = args.tdgcn_JM2d_Score
    bm_file3 = args.tdgcn_BM2d_Score
    
    j_file4 = args.mstgcn_J2d_Score
    b_file4 = args.mstgcn_B2d_Score
    jm_file4 = args.mstgcn_JM2d_Score
    bm_file4 = args.mstgcn_BM2d_Score
    
    val_txt_file = args.val_sample

    File = [j_file, b_file,jm_file, bm_file, k2_file, k2m_file,te_file, teb_file, \
            j_file2, b_file2,  jm3d_file, bm3d_file, \
            j_file3, b_file3,  jm_file3, bm_file3,\
            j_file4, b_file4, jm_file4, bm_file4]  

    if args.benchmark == 'V1':
        Numclass = 155
        Sample_Num = 4599

        Rate=[2866.446560756607, 0.0, 0.0, 0.0, 8062.399562235894, 1887.2742569619966, 
              17344.990413912543, 12994.157501998963,
              14716.495604359188, 11192.194873406577, 147.20429556949034, 0.0,
              0.0,0.0, 0.0, 0.0,
              8777.016885449711, 1212.4997536194596, 0.0, 0.0 ]

        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
        output_dir = '/media/sdd/robot/ICMEW2024-Track10/Test_dataset'
        final_score_np = final_score.numpy()

        # 保存为npy文件到/output目录
        np.save(os.path.join(output_dir, 'pred.npy'), final_score_np)
        print("集成成功！！！")

        # true_label = gen_label(val_txt_file)
    
    # Acc = Cal_Acc(final_score, true_label)

    # print('acc:', Acc)
