python ensemble_eval.py \
--mixformer_J_Score /media/sdd/robot/ICMEW2024-Track10/testB/mixformer_J_69.65.pkl \
--mixformer_B_Score /media/sdd/robot/ICMEW2024-Track10/testB/mixformer_B_63.9.pkl \
--mixformer_JM_Score /media/sdd/robot/ICMEW2024-Track10/testB/mixformer_JM_46.45.pkl \
--mixformer_BM_Score /media/sdd/robot/ICMEW2024-Track10/testB/mixformer_BM_34.2.pkl \
--tegcn_J_Score /media/sdd/robot/ICMEW2024-Track10/testB/tegcn_J_0.6985.pkl \
--tegcn_B_Score /media/sdd/robot/ICMEW2024-Track10/testB/tegcn_B_69.45%.pkl \
--mixformer_k2_Score /media/sdd/robot/ICMEW2024-Track10/testB/mixformer_k2_69.60.pkl \
--mixformer_k2M_Score /media/sdd/robot/ICMEW2024-Track10/testB/mixformer_k2m_59.55.pkl \
--ctrgcn_J2d_Score /media/sdd/robot/ICMEW2024-Track10/testB/ctrgcn-J.pkl \
--ctrgcn_B2d_Score /media/sdd/robot/ICMEW2024-Track10/testB/ctrgcn-B.pkl \
--ctrgcn_JM3d_Score /media/sdd/robot/ICMEW2024-Track10/testB/ctrgcn-JM.pkl \
--ctrgcn_BM3d_Score /media/sdd/robot/ICMEW2024-Track10/testB/ctrgcn-BM.pkl \
--tdgcn_J2d_Score /media/sdd/robot/ICMEW2024-Track10/testB/tdgcn-J.pkl \
--tdgcn_B2d_Score /media/sdd/robot/ICMEW2024-Track10/testB/tdgcn-B.pkl \
--tdgcn_JM2d_Score /media/sdd/robot/ICMEW2024-Track10/testB/tdgcn-JM.pkl \
--tdgcn_BM2d_Score /media/sdd/robot/ICMEW2024-Track10/testB/tdgcn-BM.pkl \
--mstgcn_J2d_Score /media/sdd/robot/ICMEW2024-Track10/testB/mstgcn-J.pkl \
--mstgcn_B2d_Score /media/sdd/robot/ICMEW2024-Track10/testB/mstgcn-B.pkl \
--mstgcn_JM2d_Score /media/sdd/robot/ICMEW2024-Track10/testB/mstgcn-JM.pkl \
--mstgcn_BM2d_Score /media/sdd/robot/ICMEW2024-Track10/testB/mstgcn-BM.pkl \
--val_sample ./Process_data/CS_test_V1.txt \
--benchmark V1


# 或者输入的数据只用两个通道 进行训练可以尝试
# 调整集成的权重
# 观看视频
# 集成tegcn更多的数据
