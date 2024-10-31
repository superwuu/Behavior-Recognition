python main.py --config ./test-config-B/ctrgcn_V1_J_3d.yaml --phase test --save-score True --weights output/new-ctrgcn-J/runs-56-3584.pt --device 3

python main.py --config ./test-config-B/ctrgcn_V1_JM_3d.yaml --phase test --save-score True --weights output/new-ctrgcn-JM/runs-85-5440.pt --device 3

python main.py --config ./test-config-B/mstgcn_V1_J.yaml --phase test --save-score True --weights output/new-mstgcn-J/runs-42-1344.pt --device 3

python main.py --config ./test-config-B/mstgcn_V1_JM.yaml --phase test --save-score True --weights output/new-mstgcn-JM/runs-122-3904.pt --device 3

python main.py --config ./test-config-B/tdgcn_V1_J.yaml --phase test --save-score True --weights output/new-tdgcn-J/runs-51-6528.pt --device 3

python main.py --config ./test-config-B/tdgcn_V1_JM.yaml --phase test --save-score True --weights output/new-tdgcn-JM/runs-67-4288.pt --device 3

python main.py --config ./test-config-B/ctrgcn_V1_B_3d.yaml --phase test --save-score True --weights output/new-ctrgcn-B/runs-64-4096.pt --device 0

python main.py --config ./test-config-B/ctrgcn_V1_BM_3d.yaml --phase test --save-score True --weights output/new-ctrgcn-BM_2/runs-68-4352.pt --device 0

python main.py --config ./test-config-B/mstgcn_V1_B.yaml --phase test --save-score True --weights output/new-mstgcn-B/runs-39-1248.pt --device 0

python main.py --config ./test-config-B/mstgcn_V1_BM.yaml --phase test --save-score True --weights output/mstgcn_V1_BM/runs-62-2108.pt --device 0

python main.py --config ./test-config-B/tdgcn_V1_B.yaml --phase test --save-score True --weights output/new-tdgcn-B/runs-47-6016.pt --device 0

python main.py --config ./test-config-B/tdgcn_V1_BM.yaml --phase test --save-score True --weights output/new-tdgcn-BM/runs-65-8320.pt --device 0