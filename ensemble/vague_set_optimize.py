data1=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/mixformer_J_69.65.npy")

data2=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/mixformer_B_63.9.npy")

data3=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/mixformer_JM_46.45.npy")

data4=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/mixformer_BM_34.2.npy")

data5=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/mixformer_k2_69.60.npy")

data6=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/mixformer_k2m_59.55.npy")

data7=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/tegcn_J_69.85.npy")

data8=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/tegcn_B_69.45.npy")

data9=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/ctrgcn-J.npy")

data10=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/ctrgcn-B.npy")

data11=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/ctrgcn-JM.npy")

data12=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/ctrgcn-BM.npy")

data13=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/mstgcn-J.npy")

data14=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/mstgcn-B.npy")

data15=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/mstgcn-JM.npy")

data16=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/mstgcn-BM.npy")

data17=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/tdgcn-J.npy")

data18=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/tdgcn-B.npy")

data19=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/tdgcn-JM.npy")

data20=np.load("/media/sdd/hsj/0competiton/Human_activity_recognition/1028res/tdgcn-BM.npy")


#通过训练得到的各个模型的权重超参数
Rate=[2866.446560756607, 0.0, 0.0, 0.0, 8062.399562235894, 
1887.2742569619966, 17344.990413912543, 12994.157501998963, 14716.495604359188, 11192.194873406577, 
147.20429556949034, 0.0, 0.0, 0.0,0.0, 
0.0, 8777.016885449711, 1212.4997536194596, 0.0, 0.0 ]

data=data1*Rate[0]+data2*Rate[1]+data3*Rate[2]+data4*Rate[3]+data5*Rate[4]+data6*Rate[5]+data7*Rate[6]+data8*Rate[7]+data9*Rate[8]+data10*Rate[9]
data=data+data11*Rate[10]+data12*Rate[11]+data13*Rate[12]+data14*Rate[13]+data15*Rate[14]+data16*Rate[15]+data17*Rate[16]+data18*Rate[17]+data19*Rate[18]+data20*Rate[19]

#模糊样本优化过程
# 读取CSV文件
file_path = '/media/sdd/hsj/0competiton/Human_activity_recognition/final_merged_data_select.csv'  # 文件路径
df = pd.read_csv(file_path)

file=[data10,data12,data9,data11,data4,data2,data3,data1,data5,data6,data14,data16,data13,data15,data18,data20,data17,data19,data8,data7]
lr=sum(Rate)/20

for index, row in df.iterrows():
    # 获取第2到21列的数据
    values = row[1:21].values.tolist()
    #print(row[0])
    res_ind = int(row[0])
    
    # 计算众数
    mode_result = stats.mode(values)
    #print(mode_result.mode)
    mode_value = mode_result.mode #得到众数
    #print(mode_value)
    #print(values)
    ind = values.index(mode_value)  #找到第一个数值为众数的位置(此处是为了将模糊集的判别动作替换为更正后的动作)
    data[res_ind] = file[ind][res_ind]*lr  #得到更正后的执行度

    np.save('/media/sdd/hsj/0competiton/Human_activity_recognition/pred.npy', data)

