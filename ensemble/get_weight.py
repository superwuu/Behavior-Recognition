import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import random
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


def get_weights(model_name,label_path,threadshold,topk=1):
    with open(model_name, 'rb') as f:
        data=pickle.load(f)

    values_list = list(data.values())   # 二维列表，每个样本属于每个类的概率

    max_indices = np.argmax(values_list, axis=1)

    # 求top1、top3、top5列表
    top1 = max_indices.reshape(-1, 1)
    # print(top1[0])

    top3 = np.argsort(values_list, axis=1)[:, -3:]
    # print(top3[0])

    top5 = np.argsort(values_list, axis=1)[:, -5:]
    # print(top5[0])

    array_data = np.load(label_path)
    list_data = array_data.tolist()

    num_all=[0]*155
    num_right=[0]*155

    for i,val in enumerate(list_data):
        num_all[val]+=1
        # 如果样本的标签在top里面，就认为预测正确
        if topk==1:
            if val in top1[i]:
                num_right[val]+=1
        elif topk==3:
            if val in top3[i]:
                num_right[val]+=1
        elif topk==5:
            if val in top5[i]:
                num_right[val]+=1

    res=[0.0]*155
    for i,val in enumerate(num_right):  # res: 155个标签的准确率，[155,1]
        res[i]=val/num_all[i]

    final_res = [num for num in res if num >= threadshold]

    # print(len(res)-len(final_res))

    # 计算方差
    variance_sum = sum((x - (sum(res)/len(res))) ** 2 for x in res)
    variance = variance_sum / (len(res) - 1)

    return sum(final_res)/len(final_res), sum(res)/len(res), variance

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
    
    
    File = [
            j_file, b_file,
            jm_file, bm_file,
            k2_file, k2m_file,
            te_file,teb_file,
            
            j_file2, b_file2,
            jm3d_file, bm3d_file,
            
            j_file4, b_file4,
            jm_file4, bm_file4,
            
            j_file3, b_file3,
            jm_file3, bm_file3
            ]  

    if args.benchmark == 'V1':
        Numclass = 155
        Sample_Num = 2000

        Rate=[]
        
        # Rate=[2866.446560756607, 0.0, 0.0, 0.0, 8062.399562235894, 1887.2742569619966, 
        #       17344.990413912543, 12994.157501998963,
        #       14716.495604359188, 11192.194873406577, 147.20429556949034, 0.0,
        #       0.0,0.0, 0.0, 0.0,
        #       8777.016885449711, 1212.4997536194596, 0.0, 0.0 ]
        
        max_score = 0
        best_x, best_y, best_z = 0, 0, 0
        for x in range(0, 15):
            for y in range(0, 15):
                for z in range(0, 15):
                    Rate.clear()
                    for file in File:
                        r,w,v=get_weights(file,val_txt_file,0.15,1)     # r:去掉低于阈值后的均值    w:所有类的均值      v:标准差
                        score=(r ** x) * (w ** y) / (v ** z)
                        Rate.append(score)

                    final_score = Cal_Score(File, Rate, Sample_Num, Numclass)

                    true_label = gen_label(val_txt_file)
                
                    Acc = Cal_Acc(final_score, true_label)
                    
                    if Acc > max_score:
                        max_score = Acc
                        best_x, best_y, best_z = x, y, z
            
    print(f"最高得分为: {max_score}")
    print(f"对应的 x, y, z 值为: x={best_x}, y={best_y}, z={best_z}")

    while True:
        Rate2=[]
        for it in Rate:
            tmp=random.uniform(0.0, 20000.0)
            Rate2.append(tmp)
        final_score = Cal_Score(File, Rate2, Sample_Num, Numclass)

        Acc = Cal_Acc(final_score, true_label)
        
        if Acc > max_score:
            max_score=Acc
            with open('best-new.txt', 'a') as f:
                f.write(str(max_score)+' ')
                for item in Rate2:
                    f.write(str(item) + ' ')
                f.write('\n')
                    
    #     Rate.clear()
    #     for file in File:
    #         r,w,v=get_weights(file,val_txt_file,0.15,1)     # r:去掉低于阈值后的均值    w:所有类的均值      v:标准差
    #         Rate.append(r*r*r*r*r*r*r*r*r*w*w*w*w*w*w*w/v/v/v/v/v/v/v/v/v/v)

    #     with open('my_file.txt', 'a') as f:
    #         for item in Rate:
    #             # 写入每个元素，并添加换行符
    #             f.write(str(item) + ' ')
    #         f.write('\n')


    #     final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
    #     final_score_np = final_score.numpy()
    #     true_label = gen_label(val_txt_file)
    
    # Acc = Cal_Acc(final_score, true_label)

    # print('acc:', Acc)
