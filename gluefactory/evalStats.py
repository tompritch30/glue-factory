import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.image as mpimg

log_data = """
 700] loss {total 1.054E+00, last 4.888E-01, assignment_nll 4.888E-01, nll_pos 7.374E-01, nll_neg 2.402E-01, num_matchable 1.391E+02, num_unmatchable 3.586E+02, confidence 2.294E-01, row_norm 9.201E-01}
[06/17/2024 01:17:17 gluefactory INFO] [E 4 | it 800] loss {total 8.366E-01, last 2.679E-01, assignment_nll 2.679E-01, nll_pos 2.791E-01, nll_neg 2.566E-01, num_matchable 1.356E+02, num_unmatchable 3.602E+02, confidence 2.257E-01, row_norm 9.461E-01}
[06/17/2024 01:19:52 gluefactory INFO] [E 4 | it 900] loss {total 9.119E-01, last 2.978E-01, assignment_nll 2.978E-01, nll_pos 3.524E-01, nll_neg 2.432E-01, num_matchable 1.267E+02, num_unmatchable 3.717E+02, confidence 2.317E-01, row_norm 9.357E-01}
[06/17/2024 01:22:26 gluefactory INFO] [E 4 | it 1000] loss {total 8.192E-01, last 2.965E-01, assignment_nll 2.965E-01, nll_pos 3.611E-01, nll_neg 2.320E-01, num_matchable 1.474E+02, num_unmatchable 3.485E+02, confidence 2.301E-01, row_norm 9.285E-01}
[06/17/2024 01:31:39 gluefactory INFO] [Validation] {match_recall 9.163E-01, match_precision 6.958E-01, accuracy 9.144E-01, average_precision 6.496E-01, loss/total 3.515E-01, loss/last 3.515E-01, loss/assignment_nll 3.515E-01, loss/nll_pos 4.645E-01, loss/nll_neg 2.386E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.324E-01}
[06/17/2024 01:31:39 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 01:31:41 gluefactory INFO] New best val: loss/total=0.3515352072299613
[06/17/2024 01:34:16 gluefactory INFO] [E 4 | it 1100] loss {total 9.237E-01, last 3.570E-01, assignment_nll 3.570E-01, nll_pos 4.121E-01, nll_neg 3.019E-01, num_matchable 1.384E+02, num_unmatchable 3.594E+02, confidence 2.362E-01, row_norm 9.350E-01}
[06/17/2024 01:36:51 gluefactory INFO] [E 4 | it 1200] loss {total 8.191E-01, last 3.087E-01, assignment_nll 3.087E-01, nll_pos 3.676E-01, nll_neg 2.499E-01, num_matchable 1.317E+02, num_unmatchable 3.658E+02, confidence 2.213E-01, row_norm 9.378E-01}
[06/17/2024 01:39:25 gluefactory INFO] [E 4 | it 1300] loss {total 9.793E-01, last 3.787E-01, assignment_nll 3.787E-01, nll_pos 5.151E-01, nll_neg 2.423E-01, num_matchable 1.294E+02, num_unmatchable 3.670E+02, confidence 2.275E-01, row_norm 9.329E-01}
[06/17/2024 01:41:59 gluefactory INFO] [E 4 | it 1400] loss {total 8.470E-01, last 3.044E-01, assignment_nll 3.044E-01, nll_pos 3.458E-01, nll_neg 2.630E-01, num_matchable 1.158E+02, num_unmatchable 3.801E+02, confidence 2.273E-01, row_norm 9.394E-01}
[06/17/2024 01:44:33 gluefactory INFO] [E 4 | it 1500] loss {total 1.132E+00, last 4.357E-01, assignment_nll 4.357E-01, nll_pos 5.909E-01, nll_neg 2.806E-01, num_matchable 1.142E+02, num_unmatchable 3.818E+02, confidence 2.336E-01, row_norm 9.313E-01}
[06/17/2024 01:53:37 gluefactory INFO] [Validation] {match_recall 9.163E-01, match_precision 6.934E-01, accuracy 9.139E-01, average_precision 6.474E-01, loss/total 3.584E-01, loss/last 3.584E-01, loss/assignment_nll 3.584E-01, loss/nll_pos 4.866E-01, loss/nll_neg 2.302E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.385E-01}
[06/17/2024 01:56:13 gluefactory INFO] [E 4 | it 1600] loss {total 9.513E-01, last 3.299E-01, assignment_nll 3.299E-01, nll_pos 4.233E-01, nll_neg 2.366E-01, num_matchable 1.276E+02, num_unmatchable 3.666E+02, confidence 2.310E-01, row_norm 9.324E-01}
[06/17/2024 01:58:47 gluefactory INFO] [E 4 | it 1700] loss {total 8.853E-01, last 3.337E-01, assignment_nll 3.337E-01, nll_pos 4.409E-01, nll_neg 2.266E-01, num_matchable 1.421E+02, num_unmatchable 3.532E+02, confidence 2.262E-01, row_norm 9.276E-01}
[06/17/2024 02:01:22 gluefactory INFO] [E 4 | it 1800] loss {total 8.071E-01, last 2.557E-01, assignment_nll 2.557E-01, nll_pos 2.600E-01, nll_neg 2.515E-01, num_matchable 1.472E+02, num_unmatchable 3.484E+02, confidence 2.386E-01, row_norm 9.414E-01}
[06/17/2024 02:10:53 gluefactory INFO] [Validation] {match_recall 9.240E-01, match_precision 7.051E-01, accuracy 9.188E-01, average_precision 6.626E-01, loss/total 3.217E-01, loss/last 3.217E-01, loss/assignment_nll 3.217E-01, loss/nll_pos 4.098E-01, loss/nll_neg 2.336E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.424E-01}
[06/17/2024 02:10:53 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 02:10:55 gluefactory INFO] New best val: loss/total=0.3216888875173435
[06/17/2024 02:10:57 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_4_9119.tar
[06/17/2024 02:10:58 gluefactory INFO] Starting epoch 5
[06/17/2024 02:10:58 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/17/2024 02:11:06 gluefactory INFO] [E 5 | it 0] loss {total 8.482E-01, last 3.204E-01, assignment_nll 3.204E-01, nll_pos 3.715E-01, nll_neg 2.694E-01, num_matchable 1.283E+02, num_unmatchable 3.701E+02, confidence 2.209E-01, row_norm 9.438E-01}
[06/17/2024 02:13:42 gluefactory INFO] [E 5 | it 100] loss {total 1.007E+00, last 4.662E-01, assignment_nll 4.662E-01, nll_pos 6.580E-01, nll_neg 2.743E-01, num_matchable 1.167E+02, num_unmatchable 3.824E+02, confidence 2.165E-01, row_norm 9.305E-01}
[06/17/2024 02:16:16 gluefactory INFO] [E 5 | it 200] loss {total 8.619E-01, last 2.821E-01, assignment_nll 2.821E-01, nll_pos 3.515E-01, nll_neg 2.127E-01, num_matchable 1.274E+02, num_unmatchable 3.703E+02, confidence 2.190E-01, row_norm 9.438E-01}
[06/17/2024 02:18:51 gluefactory INFO] [E 5 | it 300] loss {total 8.147E-01, last 2.354E-01, assignment_nll 2.354E-01, nll_pos 2.600E-01, nll_neg 2.108E-01, num_matchable 1.376E+02, num_unmatchable 3.594E+02, confidence 2.225E-01, row_norm 9.448E-01}
[06/17/2024 02:21:25 gluefactory INFO] [E 5 | it 400] loss {total 9.044E-01, last 2.951E-01, assignment_nll 2.951E-01, nll_pos 3.592E-01, nll_neg 2.311E-01, num_matchable 1.226E+02, num_unmatchable 3.704E+02, confidence 2.413E-01, row_norm 9.452E-01}
[06/17/2024 02:23:59 gluefactory INFO] [E 5 | it 500] loss {total 9.274E-01, last 3.512E-01, assignment_nll 3.512E-01, nll_pos 4.652E-01, nll_neg 2.372E-01, num_matchable 1.272E+02, num_unmatchable 3.689E+02, confidence 2.369E-01, row_norm 9.292E-01}
[06/17/2024 02:32:55 gluefactory INFO] [Validation] {match_recall 9.266E-01, match_precision 7.092E-01, accuracy 9.206E-01, average_precision 6.678E-01, loss/total 3.082E-01, loss/last 3.082E-01, loss/assignment_nll 3.082E-01, loss/nll_pos 3.954E-01, loss/nll_neg 2.210E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.413E-01}
[06/17/2024 02:32:55 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 02:32:57 gluefactory INFO] New best val: loss/total=0.30815616240647087
[06/17/2024 02:35:33 gluefactory INFO] [E 5 | it 600] loss {total 1.140E+00, last 5.531E-01, assignment_nll 5.531E-01, nll_pos 8.022E-01, nll_neg 3.039E-01, num_matchable 1.185E+02, num_unmatchable 3.791E+02, confidence 2.332E-01, row_norm 9.100E-01}
[06/17/2024 02:38:07 gluefactory INFO] [E 5 | it 700] loss {total 9.649E-01, last 4.449E-01, assignment_nll 4.449E-01, nll_pos 6.009E-01, nll_neg 2.889E-01, num_matchable 1.232E+02, num_unmatchable 3.743E+02, confidence 2.125E-01, row_norm 9.006E-01}
[06/17/2024 02:40:41 gluefactory INFO] [E 5 | it 800] loss {total 8.352E-01, last 2.925E-01, assignment_nll 2.925E-01, nll_pos 3.661E-01, nll_neg 2.189E-01, num_matchable 1.308E+02, num_unmatchable 3.631E+02, confidence 2.249E-01, row_norm 9.473E-01}
[06/17/2024 02:42:44 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_5_10000.tar
[06/17/2024 02:43:17 gluefactory INFO] [E 5 | it 900] loss {total 9.936E-01, last 4.489E-01, assignment_nll 4.489E-01, nll_pos 6.317E-01, nll_neg 2.662E-01, num_matchable 1.184E+02, num_unmatchable 3.779E+02, confidence 2.171E-01, row_norm 9.348E-01}
[06/17/2024 02:45:50 gluefactory INFO] [E 5 | it 1000] loss {total 9.403E-01, last 4.270E-01, assignment_nll 4.270E-01, nll_pos 6.136E-01, nll_neg 2.404E-01, num_matchable 1.456E+02, num_unmatchable 3.462E+02, confidence 2.168E-01, row_norm 9.234E-01}
[06/17/2024 02:54:42 gluefactory INFO] [Validation] {match_recall 9.299E-01, match_precision 7.184E-01, accuracy 9.250E-01, average_precision 6.780E-01, loss/total 2.938E-01, loss/last 2.938E-01, loss/assignment_nll 2.938E-01, loss/nll_pos 3.801E-01, loss/nll_neg 2.075E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.419E-01}
[06/17/2024 02:54:42 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 02:54:43 gluefactory INFO] New best val: loss/total=0.29377067208306434
[06/17/2024 02:57:19 gluefactory INFO] [E 5 | it 1100] loss {total 9.335E-01, last 3.297E-01, assignment_nll 3.297E-01, nll_pos 4.008E-01, nll_neg 2.586E-01, num_matchable 1.417E+02, num_unmatchable 3.556E+02, confidence 2.443E-01, row_norm 9.468E-01}
[06/17/2024 02:59:53 gluefactory INFO] [E 5 | it 1200] loss {total 8.738E-01, last 2.947E-01, assignment_nll 2.947E-01, nll_pos 3.898E-01, nll_neg 1.997E-01, num_matchable 1.379E+02, num_unmatchable 3.589E+02, confidence 2.259E-01, row_norm 9.391E-01}
[06/17/2024 03:02:29 gluefactory INFO] [E 5 | it 1300] loss {total 9.819E-01, last 3.745E-01, assignment_nll 3.745E-01, nll_pos 4.733E-01, nll_neg 2.757E-01, num_matchable 1.312E+02, num_unmatchable 3.644E+02, confidence 2.348E-01, row_norm 9.125E-01}
[06/17/2024 03:05:06 gluefactory INFO] [E 5 | it 1400] loss {total 1.023E+00, last 4.190E-01, assignment_nll 4.190E-01, nll_pos 6.047E-01, nll_neg 2.333E-01, num_matchable 1.172E+02, num_unmatchable 3.801E+02, confidence 2.262E-01, row_norm 9.232E-01}
[06/17/2024 03:07:39 gluefactory INFO] [E 5 | it 1500] loss {total 8.611E-01, last 2.548E-01, assignment_nll 2.548E-01, nll_pos 2.978E-01, nll_neg 2.118E-01, num_matchable 1.220E+02, num_unmatchable 3.741E+02, confidence 2.273E-01, row_norm 9.495E-01}
[06/17/2024 03:16:30 gluefactory INFO] [Validation] {match_recall 9.288E-01, match_precision 7.139E-01, accuracy 9.232E-01, average_precision 6.740E-01, loss/total 3.052E-01, loss/last 3.052E-01, loss/assignment_nll 3.052E-01, loss/nll_pos 4.041E-01, loss/nll_neg 2.062E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.413E-01}
[06/17/2024 03:19:05 gluefactory INFO] [E 5 | it 1600] loss {total 8.417E-01, last 2.481E-01, assignment_nll 2.481E-01, nll_pos 2.967E-01, nll_neg 1.995E-01, num_matchable 1.253E+02, num_unmatchable 3.703E+02, confidence 2.202E-01, row_norm 9.435E-01}
[06/17/2024 03:21:40 gluefactory INFO] [E 5 | it 1700] loss {total 7.578E-01, last 2.141E-01, assignment_nll 2.141E-01, nll_pos 2.444E-01, nll_neg 1.838E-01, num_matchable 1.383E+02, num_unmatchable 3.592E+02, confidence 2.158E-01, row_norm 9.427E-01}
[06/17/2024 03:24:14 gluefactory INFO] [E 5 | it 1800] loss {total 7.610E-01, last 2.269E-01, assignment_nll 2.269E-01, nll_pos 2.497E-01, nll_neg 2.042E-01, num_matchable 1.353E+02, num_unmatchable 3.614E+02, confidence 2.253E-01, row_norm 9.556E-01}
[06/17/2024 03:33:48 gluefactory INFO] [Validation] {match_recall 9.264E-01, match_precision 7.227E-01, accuracy 9.266E-01, average_precision 6.809E-01, loss/total 3.008E-01, loss/last 3.008E-01, loss/assignment_nll 3.008E-01, loss/nll_pos 3.950E-01, loss/nll_neg 2.065E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.399E-01}
[06/17/2024 03:33:50 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_5_10943.tar
[06/17/2024 03:33:52 gluefactory INFO] Starting epoch 6
[06/17/2024 03:33:52 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/17/2024 03:34:00 gluefactory INFO] [E 6 | it 0] loss {total 8.962E-01, last 3.363E-01, assignment_nll 3.363E-01, nll_pos 4.296E-01, nll_neg 2.429E-01, num_matchable 1.248E+02, num_unmatchable 3.731E+02, confidence 2.202E-01, row_norm 9.262E-01}
[06/17/2024 03:36:35 gluefactory INFO] [E 6 | it 100] loss {total 9.095E-01, last 3.350E-01, assignment_nll 3.350E-01, nll_pos 4.333E-01, nll_neg 2.367E-01, num_matchable 1.131E+02, num_unmatchable 3.849E+02, confidence 2.194E-01, row_norm 9.405E-01}
[06/17/2024 03:39:09 gluefactory INFO] [E 6 | it 200] loss {total 8.480E-01, last 2.984E-01, assignment_nll 2.984E-01, nll_pos 3.776E-01, nll_neg 2.191E-01, num_matchable 1.131E+02, num_unmatchable 3.850E+02, confidence 2.064E-01, row_norm 9.347E-01}
[06/17/2024 03:41:44 gluefactory INFO] [E 6 | it 300] loss {total 8.830E-01, last 2.787E-01, assignment_nll 2.787E-01, nll_pos 3.666E-01, nll_neg 1.909E-01, num_matchable 1.422E+02, num_unmatchable 3.543E+02, confidence 2.230E-01, row_norm 9.435E-01}
[06/17/2024 03:44:18 gluefactory INFO] [E 6 | it 400] loss {total 9.386E-01, last 3.367E-01, assignment_nll 3.367E-01, nll_pos 4.569E-01, nll_neg 2.165E-01, num_matchable 1.323E+02, num_unmatchable 3.611E+02, confidence 2.351E-01, row_norm 9.341E-01}
[06/17/2024 03:46:52 gluefactory INFO] [E 6 | it 500] loss {total 9.494E-01, last 3.634E-01, assignment_nll 3.634E-01, nll_pos 5.017E-01, nll_neg 2.252E-01, num_matchable 1.155E+02, num_unmatchable 3.817E+02, confidence 2.252E-01, row_norm 9.277E-01}
[06/17/2024 03:55:48 gluefactory INFO] [Validation] {match_recall 9.329E-01, match_precision 7.198E-01, accuracy 9.259E-01, average_precision 6.810E-01, loss/total 2.856E-01, loss/last 2.856E-01, loss/assignment_nll 2.856E-01, loss/nll_pos 3.621E-01, loss/nll_neg 2.091E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.457E-01}
[06/17/2024 03:55:48 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 03:55:50 gluefactory INFO] New best val: loss/total=0.28555652990392144
[06/17/2024 03:58:26 gluefactory INFO] [E 6 | it 600] loss {total 7.424E-01, last 2.202E-01, assignment_nll 2.202E-01, nll_pos 2.169E-01, nll_neg 2.236E-01, num_matchable 1.231E+02, num_unmatchable 3.757E+02, confidence 2.168E-01, row_norm 9.515E-01}
[06/17/2024 04:01:00 gluefactory INFO] [E 6 | it 700] loss {total 8.192E-01, last 2.451E-01, assignment_nll 2.451E-01, nll_pos 2.982E-01, nll_neg 1.920E-01, num_matchable 1.218E+02, num_unmatchable 3.759E+02, confidence 2.106E-01, row_norm 9.489E-01}
[06/17/2024 04:03:34 gluefactory INFO] [E 6 | it 800] loss {total 8.894E-01, last 3.611E-01, assignment_nll 3.611E-01, nll_pos 5.159E-01, nll_neg 2.064E-01, num_matchable 1.200E+02, num_unmatchable 3.756E+02, confidence 2.100E-01, row_norm 9.421E-01}
[06/17/2024 04:06:08 gluefactory INFO] [E 6 | it 900] loss {total 7.546E-01, last 2.197E-01, assignment_nll 2.197E-01, nll_pos 2.475E-01, nll_neg 1.920E-01, num_matchable 1.258E+02, num_unmatchable 3.730E+02, confidence 2.058E-01, row_norm 9.589E-01}
[06/17/2024 04:08:43 gluefactory INFO] [E 6 | it 1000] loss {total 7.588E-01, last 2.379E-01, assignment_nll 2.379E-01, nll_pos 2.936E-01, nll_neg 1.823E-01, num_matchable 1.533E+02, num_unmatchable 3.416E+02, confidence 2.227E-01, row_norm 9.470E-01}
[06/17/2024 04:17:51 gluefactory INFO] [Validation] {match_recall 9.344E-01, match_precision 7.263E-01, accuracy 9.286E-01, average_precision 6.887E-01, loss/total 2.722E-01, loss/last 2.722E-01, loss/assignment_nll 2.722E-01, loss/nll_pos 3.478E-01, loss/nll_neg 1.967E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.460E-01}
[06/17/2024 04:17:51 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 04:17:52 gluefactory INFO] New best val: loss/total=0.2722219196569921
[06/17/2024 04:20:28 gluefactory INFO] [E 6 | it 1100] loss {total 8.332E-01, last 2.261E-01, assignment_nll 2.261E-01, nll_pos 2.511E-01, nll_neg 2.010E-01, num_matchable 1.297E+02, num_unmatchable 3.693E+02, confidence 2.313E-01, row_norm 9.593E-01}
[06/17/2024 04:23:02 gluefactory INFO] [E 6 | it 1200] loss {total 7.921E-01, last 2.711E-01, assignment_nll 2.711E-01, nll_pos 3.389E-01, nll_neg 2.033E-01, num_matchable 1.324E+02, num_unmatchable 3.649E+02, confidence 2.195E-01, row_norm 9.481E-01}
[06/17/2024 04:25:36 gluefactory INFO] [E 6 | it 1300] loss {total 8.390E-01, last 2.546E-01, assignment_nll 2.546E-01, nll_pos 3.209E-01, nll_neg 1.883E-01, num_matchable 1.316E+02, num_unmatchable 3.642E+02, confidence 2.230E-01, row_norm 9.472E-01}
[06/17/2024 04:28:10 gluefactory INFO] [E 6 | it 1400] loss {total 9.134E-01, last 3.502E-01, assignment_nll 3.502E-01, nll_pos 5.121E-01, nll_neg 1.883E-01, num_matchable 1.074E+02, num_unmatchable 3.908E+02, confidence 2.000E-01, row_norm 9.462E-01}
[06/17/2024 04:30:44 gluefactory INFO] [E 6 | it 1500] loss {total 1.089E+00, last 3.849E-01, assignment_nll 3.849E-01, nll_pos 5.515E-01, nll_neg 2.182E-01, num_matchable 1.140E+02, num_unmatchable 3.831E+02, confidence 2.216E-01, row_norm 9.318E-01}
[06/17/2024 04:39:45 gluefactory INFO] [Validation] {match_recall 9.388E-01, match_precision 7.404E-01, accuracy 9.348E-01, average_precision 7.038E-01, loss/total 2.497E-01, loss/last 2.497E-01, loss/assignment_nll 2.497E-01, loss/nll_pos 3.209E-01, loss/nll_neg 1.785E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.533E-01}
[06/17/2024 04:39:45 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 04:39:46 gluefactory INFO] New best val: loss/total=0.24973691118990074
[06/17/2024 04:42:22 gluefactory INFO] [E 6 | it 1600] loss {total 7.097E-01, last 1.690E-01, assignment_nll 1.690E-01, nll_pos 1.738E-01, nll_neg 1.642E-01, num_matchable 1.234E+02, num_unmatchable 3.738E+02, confidence 2.029E-01, row_norm 9.631E-01}
[06/17/2024 04:44:56 gluefactory INFO] [E 6 | it 1700] loss {total 7.787E-01, last 2.626E-01, assignment_nll 2.626E-01, nll_pos 3.596E-01, nll_neg 1.657E-01, num_matchable 1.419E+02, num_unmatchable 3.541E+02, confidence 2.072E-01, row_norm 9.418E-01}
[06/17/2024 04:47:31 gluefactory INFO] [E 6 | it 1800] loss {total 7.344E-01, last 1.828E-01, assignment_nll 1.828E-01, nll_pos 1.893E-01, nll_neg 1.763E-01, num_matchable 1.306E+02, num_unmatchable 3.675E+02, confidence 2.226E-01, row_norm 9.587E-01}
[06/17/2024 04:57:01 gluefactory INFO] [Validation] {match_recall 9.374E-01, match_precision 7.408E-01, accuracy 9.347E-01, average_precision 7.035E-01, loss/total 2.573E-01, loss/last 2.573E-01, loss/assignment_nll 2.573E-01, loss/nll_pos 3.333E-01, loss/nll_neg 1.813E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.538E-01}
[06/17/2024 04:57:02 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_6_12767.tar
[06/17/2024 04:57:04 gluefactory INFO] Starting epoch 7
[06/17/2024 04:57:04 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/17/2024 04:57:13 gluefactory INFO] [E 7 | it 0] loss {total 8.586E-01, last 2.657E-01, assignment_nll 2.657E-01, nll_pos 3.183E-01, nll_neg 2.130E-01, num_matchable 1.239E+02, num_unmatchable 3.733E+02, confidence 2.269E-01, row_norm 9.477E-01}
[06/17/2024 04:59:48 gluefactory INFO] [E 7 | it 100] loss {total 7.850E-01, last 2.309E-01, assignment_nll 2.309E-01, nll_pos 2.704E-01, nll_neg 1.915E-01, num_matchable 1.203E+02, num_unmatchable 3.769E+02, confidence 2.252E-01, row_norm 9.611E-01}
[06/17/2024 05:02:22 gluefactory INFO] [E 7 | it 200] loss {total 9.234E-01, last 3.538E-01, assignment_nll 3.538E-01, nll_pos 5.265E-01, nll_neg 1.811E-01, num_matchable 1.185E+02, num_unmatchable 3.795E+02, confidence 2.177E-01, row_norm 9.425E-01}
[06/17/2024 05:04:56 gluefactory INFO] [E 7 | it 300] loss {total 6.914E-01, last 1.621E-01, assignment_nll 1.621E-01, nll_pos 1.833E-01, nll_neg 1.408E-01, num_matchable 1.470E+02, num_unmatchable 3.498E+02, confidence 2.085E-01, row_norm 9.606E-01}
[06/17/2024 05:07:30 gluefactory INFO] [E 7 | it 400] loss {total 7.357E-01, last 1.990E-01, assignment_nll 1.990E-01, nll_pos 2.334E-01, nll_neg 1.646E-01, num_matchable 1.185E+02, num_unmatchable 3.767E+02, confidence 2.114E-01, row_norm 9.557E-01}
[06/17/2024 05:10:04 gluefactory INFO] [E 7 | it 500] loss {total 7.883E-01, last 2.950E-01, assignment_nll 2.950E-01, nll_pos 4.159E-01, nll_neg 1.741E-01, num_matchable 1.221E+02, num_unmatchable 3.771E+02, confidence 1.995E-01, row_norm 9.501E-01}
[06/17/2024 05:19:01 gluefactory INFO] [Validation] {match_recall 9.416E-01, match_precision 7.418E-01, accuracy 9.354E-01, average_precision 7.068E-01, loss/total 2.462E-01, loss/last 2.462E-01, loss/assignment_nll 2.462E-01, loss/nll_pos 3.217E-01, loss/nll_neg 1.706E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.545E-01}
[06/17/2024 05:19:01 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 05:19:02 gluefactory INFO] New best val: loss/total=0.24615911035149665
[06/17/2024 05:21:38 gluefactory INFO] [E 7 | it 600] loss {total 9.388E-01, last 2.842E-01, assignment_nll 2.842E-01, nll_pos 3.470E-01, nll_neg 2.213E-01, num_matchable 1.233E+02, num_unmatchable 3.747E+02, confidence 2.384E-01, row_norm 9.435E-01}
[06/17/2024 05:24:12 gluefactory INFO] [E 7 | it 700] loss {total 9.103E-01, last 3.377E-01, assignment_nll 3.377E-01, nll_pos 4.980E-01, nll_neg 1.775E-01, num_matchable 1.381E+02, num_unmatchable 3.594E+02, confidence 2.044E-01, row_norm 9.339E-01}
[06/17/2024 05:26:46 gluefactory INFO] [E 7 | it 800] loss {total 7.883E-01, last 2.247E-01, assignment_nll 2.247E-01, nll_pos 2.763E-01, nll_neg 1.730E-01, num_matchable 1.284E+02, num_unmatchable 3.660E+02, confidence 2.206E-01, row_norm 9.474E-01}
[06/17/2024 05:29:20 gluefactory INFO] [E 7 | it 900] loss {total 9.514E-01, last 2.991E-01, assignment_nll 2.991E-01, nll_pos 3.853E-01, nll_neg 2.130E-01, num_matchable 1.055E+02, num_unmatchable 3.908E+02, confidence 2.233E-01, row_norm 9.438E-01}
[06/17/2024 05:31:56 gluefactory INFO] [E 7 | it 1000] loss {total 7.033E-01, last 1.787E-01, assignment_nll 1.787E-01, nll_pos 2.077E-01, nll_neg 1.497E-01, num_matchable 1.491E+02, num_unmatchable 3.452E+02, confidence 2.149E-01, row_norm 9.498E-01}
[06/17/2024 05:40:47 gluefactory INFO] [Validation] {match_recall 9.391E-01, match_precision 7.473E-01, accuracy 9.374E-01, average_precision 7.110E-01, loss/total 2.480E-01, loss/last 2.480E-01, loss/assignment_nll 2.480E-01, loss/nll_pos 3.325E-01, loss/nll_neg 1.635E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.496E-01}
[06/17/2024 05:43:22 gluefactory INFO] [E 7 | it 1100] loss {total 6.762E-01, last 1.663E-01, assignment_nll 1.663E-01, nll_pos 1.568E-01, nll_neg 1.758E-01, num_matchable 1.312E+02, num_unmatchable 3.675E+02, confidence 2.102E-01, row_norm 9.643E-01}
[06/17/2024 05:45:57 gluefactory INFO] [E 7 | it 1200] loss {total 8.025E-01, last 2.421E-01, assignment_nll 2.421E-01, nll_pos 2.984E-01, nll_neg 1.857E-01, num_matchable 1.337E+02, num_unmatchable 3.626E+02, confidence 2.252E-01, row_norm 9.503E-01}
[06/17/2024 05:48:31 gluefactory INFO] [E 7 | it 1300] loss {total 8.213E-01, last 2.336E-01, assignment_nll 2.336E-01, nll_pos 3.025E-01, nll_neg 1.646E-01, num_matchable 1.429E+02, num_unmatchable 3.534E+02, confidence 2.155E-01, row_norm 9.512E-01}
[06/17/2024 05:51:04 gluefactory INFO] [E 7 | it 1400] loss {total 8.823E-01, last 2.563E-01, assignment_nll 2.563E-01, nll_pos 3.278E-01, nll_neg 1.848E-01, num_matchable 1.178E+02, num_unmatchable 3.789E+02, confidence 2.262E-01, row_norm 9.429E-01}
[06/17/2024 05:53:38 gluefactory INFO] [E 7 | it 1500] loss {total 8.985E-01, last 2.655E-01, assignment_nll 2.655E-01, nll_pos 3.391E-01, nll_neg 1.920E-01, num_matchable 1.127E+02, num_unmatchable 3.832E+02, confidence 2.312E-01, row_norm 9.408E-01}
[06/17/2024 06:02:31 gluefactory INFO] [Validation] {match_recall 9.378E-01, match_precision 7.419E-01, accuracy 9.355E-01, average_precision 7.057E-01, loss/total 2.551E-01, loss/last 2.551E-01, loss/assignment_nll 2.551E-01, loss/nll_pos 3.391E-01, loss/nll_neg 1.712E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.459E-01}
[06/17/2024 06:05:07 gluefactory INFO] [E 7 | it 1600] loss {total 7.310E-01, last 1.586E-01, assignment_nll 1.586E-01, nll_pos 1.703E-01, nll_neg 1.469E-01, num_matchable 1.280E+02, num_unmatchable 3.690E+02, confidence 2.127E-01, row_norm 9.652E-01}
[06/17/2024 06:07:41 gluefactory INFO] [E 7 | it 1700] loss {total 8.056E-01, last 2.779E-01, assignment_nll 2.779E-01, nll_pos 3.925E-01, nll_neg 1.633E-01, num_matchable 1.425E+02, num_unmatchable 3.550E+02, confidence 2.062E-01, row_norm 9.449E-01}
[06/17/2024 06:10:16 gluefactory INFO] [E 7 | it 1800] loss {total 7.506E-01, last 2.004E-01, assignment_nll 2.004E-01, nll_pos 2.301E-01, nll_neg 1.706E-01, num_matchable 1.329E+02, num_unmatchable 3.617E+02, confidence 2.279E-01, row_norm 9.655E-01}
[06/17/2024 06:19:44 gluefactory INFO] [Validation] {match_recall 9.444E-01, match_precision 7.510E-01, accuracy 9.393E-01, average_precision 7.172E-01, loss/total 2.288E-01, loss/last 2.288E-01, loss/assignment_nll 2.288E-01, loss/nll_pos 2.979E-01, loss/nll_neg 1.596E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.569E-01}
[06/17/2024 06:19:44 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 06:19:45 gluefactory INFO] New best val: loss/total=0.228758949844098
[06/17/2024 06:19:47 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_7_14591.tar
[06/17/2024 06:19:48 gluefactory INFO] Starting epoch 8
[06/17/2024 06:19:48 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/17/2024 06:19:57 gluefactory INFO] [E 8 | it 0] loss {total 7.716E-01, last 2.277E-01, assignment_nll 2.277E-01, nll_pos 2.835E-01, nll_neg 1.719E-01, num_matchable 1.270E+02, num_unmatchable 3.708E+02, confidence 2.138E-01, row_norm 9.537E-01}
[06/17/2024 06:22:33 gluefactory INFO] [E 8 | it 100] loss {total 1.059E+00, last 4.893E-01, assignment_nll 4.893E-01, nll_pos 7.398E-01, nll_neg 2.389E-01, num_matchable 1.242E+02, num_unmatchable 3.721E+02, confidence 2.174E-01, row_norm 9.303E-01}
[06/17/2024 06:25:07 gluefactory INFO] [E 8 | it 200] loss {total 8.905E-01, last 3.080E-01, assignment_nll 3.080E-01, nll_pos 4.389E-01, nll_neg 1.771E-01, num_matchable 1.272E+02, num_unmatchable 3.684E+02, confidence 2.137E-01, row_norm 9.395E-01}
[06/17/2024 06:27:41 gluefactory INFO] [E 8 | it 300] loss {total 7.331E-01, last 1.917E-01, assignment_nll 1.917E-01, nll_pos 1.770E-01, nll_neg 2.064E-01, num_matchable 1.311E+02, num_unmatchable 3.658E+02, confidence 2.117E-01, row_norm 9.568E-01}
[06/17/2024 06:30:16 gluefactory INFO] [E 8 | it 400] loss {total 7.541E-01, last 2.175E-01, assignment_nll 2.175E-01, nll_pos 2.904E-01, nll_neg 1.447E-01, num_matchable 1.183E+02, num_unmatchable 3.767E+02, confidence 2.099E-01, row_norm 9.583E-01}
[06/17/2024 06:30:28 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_8_15000.tar
[06/17/2024 06:30:30 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_0_1823.tar
[06/17/2024 06:32:51 gluefactory INFO] [E 8 | it 500] loss {total 7.912E-01, last 2.250E-01, assignment_nll 2.250E-01, nll_pos 2.700E-01, nll_neg 1.801E-01, num_matchable 1.348E+02, num_unmatchable 3.616E+02, confidence 2.214E-01, row_norm 9.512E-01}
[06/17/2024 06:41:40 gluefactory INFO] [Validation] {match_recall 9.432E-01, match_precision 7.512E-01, accuracy 9.396E-01, average_precision 7.169E-01, loss/total 2.356E-01, loss/last 2.356E-01, loss/assignment_nll 2.356E-01, loss/nll_pos 3.060E-01, loss/nll_neg 1.651E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.575E-01}
[06/17/2024 06:44:16 gluefactory INFO] [E 8 | it 600] loss {total 8.178E-01, last 2.107E-01, assignment_nll 2.107E-01, nll_pos 2.269E-01, nll_neg 1.945E-01, num_matchable 1.290E+02, num_unmatchable 3.687E+02, confidence 2.171E-01, row_norm 9.518E-01}
[06/17/2024 06:46:51 gluefactory INFO] [E 8 | it 700] loss {total 9.132E-01, last 3.857E-01, assignment_nll 3.857E-01, nll_pos 5.971E-01, nll_neg 1.744E-01, num_matchable 1.399E+02, num_unmatchable 3.566E+02, confidence 2.133E-01, row_norm 9.429E-01}
[06/17/2024 06:49:26 gluefactory INFO] [E 8 | it 800] loss {total 6.740E-01, last 1.557E-01, assignment_nll 1.557E-01, nll_pos 1.590E-01, nll_neg 1.523E-01, num_matchable 1.276E+02, num_unmatchable 3.675E+02, confidence 2.086E-01, row_norm 9.638E-01}
[06/17/2024 06:52:00 gluefactory INFO] [E 8 | it 900] loss {total 7.685E-01, last 2.195E-01, assignment_nll 2.195E-01, nll_pos 2.492E-01, nll_neg 1.898E-01, num_matchable 1.209E+02, num_unmatchable 3.767E+02, confidence 2.182E-01, row_norm 9.557E-01}
[06/17/2024 06:54:35 gluefactory INFO] [E 8 | it 1000] loss {total 6.644E-01, last 1.515E-01, assignment_nll 1.515E-01, nll_pos 1.660E-01, nll_neg 1.371E-01, num_matchable 1.578E+02, num_unmatchable 3.381E+02, confidence 2.164E-01, row_norm 9.614E-01}
[06/17/2024 07:03:30 gluefactory INFO] [Validation] {match_recall 9.465E-01, match_precision 7.552E-01, accuracy 9.409E-01, average_precision 7.223E-01, loss/total 2.210E-01, loss/last 2.210E-01, loss/assignment_nll 2.210E-01, loss/nll_pos 2.807E-01, loss/nll_neg 1.613E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.564E-01}
[06/17/2024 07:03:30 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 07:03:32 gluefactory INFO] New best val: loss/total=0.22101470419341104
[06/17/2024 07:06:10 gluefactory INFO] [E 8 | it 1100] loss {total 6.610E-01, last 1.539E-01, assignment_nll 1.539E-01, nll_pos 1.778E-01, nll_neg 1.301E-01, num_matchable 1.411E+02, num_unmatchable 3.579E+02, confidence 2.094E-01, row_norm 9.659E-01}
[06/17/2024 07:08:44 gluefactory INFO] [E 8 | it 1200] loss {total 8.216E-01, last 2.422E-01, assignment_nll 2.422E-01, nll_pos 3.158E-01, nll_neg 1.686E-01, num_matchable 1.364E+02, num_unmatchable 3.608E+02, confidence 2.351E-01, row_norm 9.503E-01}
[06/17/2024 07:11:18 gluefactory INFO] [E 8 | it 1300] loss {total 1.060E+00, last 4.726E-01, assignment_nll 4.726E-01, nll_pos 6.981E-01, nll_neg 2.471E-01, num_matchable 1.279E+02, num_unmatchable 3.692E+02, confidence 2.312E-01, row_norm 9.260E-01}
[06/17/2024 07:13:53 gluefactory INFO] [E 8 | it 1400] loss {total 8.738E-01, last 2.427E-01, assignment_nll 2.427E-01, nll_pos 3.027E-01, nll_neg 1.827E-01, num_matchable 1.091E+02, num_unmatchable 3.872E+02, confidence 2.235E-01, row_norm 9.499E-01}
[06/17/2024 07:16:28 gluefactory INFO] [E 8 | it 1500] loss {total 8.746E-01, last 2.463E-01, assignment_nll 2.463E-01, nll_pos 3.007E-01, nll_neg 1.919E-01, num_matchable 1.158E+02, num_unmatchable 3.809E+02, confidence 2.206E-01, row_norm 9.548E-01}
[06/17/2024 07:25:20 gluefactory INFO] [Validation] {match_recall 9.455E-01, match_precision 7.483E-01, accuracy 9.382E-01, average_precision 7.155E-01, loss/total 2.259E-01, loss/last 2.259E-01, loss/assignment_nll 2.259E-01, loss/nll_pos 2.793E-01, loss/nll_neg 1.724E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.546E-01}
[06/17/2024 07:27:56 gluefactory INFO] [E 8 | it 1600] loss {total 6.695E-01, last 1.428E-01, assignment_nll 1.428E-01, nll_pos 1.457E-01, nll_neg 1.398E-01, num_matchable 1.318E+02, num_unmatchable 3.645E+02, confidence 2.001E-01, row_norm 9.707E-01}
[06/17/2024 07:30:30 gluefactory INFO] [E 8 | it 1700] loss {total 8.473E-01, last 3.401E-01, assignment_nll 3.401E-01, nll_pos 5.301E-01, nll_neg 1.501E-01, num_matchable 1.434E+02, num_unmatchable 3.543E+02, confidence 1.899E-01, row_norm 9.460E-01}
[06/17/2024 07:33:04 gluefactory INFO] [E 8 | it 1800] loss {total 7.491E-01, last 1.948E-01, assignment_nll 1.948E-01, nll_pos 2.205E-01, nll_neg 1.691E-01, num_matchable 1.212E+02, num_unmatchable 3.752E+02, confidence 2.220E-01, row_norm 9.596E-01}
[06/17/2024 07:42:33 gluefactory INFO] [Validation] {match_recall 9.452E-01, match_precision 7.590E-01, accuracy 9.424E-01, average_precision 7.255E-01, loss/total 2.243E-01, loss/last 2.243E-01, loss/assignment_nll 2.243E-01, loss/nll_pos 2.993E-01, loss/nll_neg 1.494E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.587E-01}
[06/17/2024 07:42:35 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_8_16415.tar
[06/17/2024 07:42:37 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_1_3647.tar
[06/17/2024 07:42:37 gluefactory INFO] Starting epoch 9
[06/17/2024 07:42:37 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/17/2024 07:42:44 gluefactory INFO] [E 9 | it 0] loss {total 7.988E-01, last 2.216E-01, assignment_nll 2.216E-01, nll_pos 2.681E-01, nll_neg 1.751E-01, num_matchable 1.245E+02, num_unmatchable 3.734E+02, confidence 2.186E-01, row_norm 9.542E-01}
[06/17/2024 07:45:19 gluefactory INFO] [E 9 | it 100] loss {total 8.720E-01, last 2.832E-01, assignment_nll 2.832E-01, nll_pos 3.754E-01, nll_neg 1.910E-01, num_matchable 1.051E+02, num_unmatchable 3.915E+02, confidence 2.170E-01, row_norm 9.463E-01}
[06/17/2024 07:47:53 gluefactory INFO] [E 9 | it 200] loss {total 7.228E-01, last 1.428E-01, assignment_nll 1.428E-01, nll_pos 1.562E-01, nll_neg 1.294E-01, num_matchable 1.342E+02, num_unmatchable 3.636E+02, confidence 2.118E-01, row_norm 9.705E-01}
[06/17/2024 07:50:27 gluefactory INFO] [E 9 | it 300] loss {total 7.633E-01, last 1.974E-01, assignment_nll 1.974E-01, nll_pos 2.483E-01, nll_neg 1.466E-01, num_matchable 1.322E+02, num_unmatchable 3.655E+02, confidence 2.080E-01, row_norm 9.524E-01}
[06/17/2024 07:53:01 gluefactory INFO] [E 9 | it 400] loss {total 6.964E-01, last 1.676E-01, assignment_nll 1.676E-01, nll_pos 1.958E-01, nll_neg 1.395E-01, num_matchable 1.311E+02, num_unmatchable 3.623E+02, confidence 2.100E-01, row_norm 9.626E-01}
[06/17/2024 07:55:36 gluefactory INFO] [E 9 | it 500] loss {total 7.053E-01, last 1.999E-01, assignment_nll 1.999E-01, nll_pos 2.508E-01, nll_neg 1.490E-01, num_matchable 1.321E+02, num_unmatchable 3.654E+02, confidence 2.175E-01, row_norm 9.574E-01}
[06/17/2024 08:04:36 gluefactory INFO] [Validation] {match_recall 9.461E-01, match_precision 7.632E-01, accuracy 9.441E-01, average_precision 7.295E-01, loss/total 2.185E-01, loss/last 2.185E-01, loss/assignment_nll 2.185E-01, loss/nll_pos 2.861E-01, loss/nll_neg 1.509E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.590E-01}
[06/17/2024 08:04:36 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 08:04:37 gluefactory INFO] New best val: loss/total=0.21850277691903972
[06/17/2024 08:07:13 gluefactory INFO] [E 9 | it 600] loss {total 7.904E-01, last 2.395E-01, assignment_nll 2.395E-01, nll_pos 3.115E-01, nll_neg 1.674E-01, num_matchable 1.199E+02, num_unmatchable 3.770E+02, confidence 2.112E-01, row_norm 9.598E-01}
[06/17/2024 08:09:47 gluefactory INFO] [E 9 | it 700] loss {total 6.833E-01, last 1.585E-01, assignment_nll 1.585E-01, nll_pos 1.873E-01, nll_neg 1.297E-01, num_matchable 1.475E+02, num_unmatchable 3.498E+02, confidence 2.055E-01, row_norm 9.681E-01}
[06/17/2024 08:12:21 gluefactory INFO] [E 9 | it 800] loss {total 6.947E-01, last 1.613E-01, assignment_nll 1.613E-01, nll_pos 1.843E-01, nll_neg 1.382E-01, num_matchable 1.253E+02, num_unmatchable 3.717E+02, confidence 2.070E-01, row_norm 9.640E-01}
[06/17/2024 08:14:55 gluefactory INFO] [E 9 | it 900] loss {total 8.216E-01, last 2.586E-01, assignment_nll 2.586E-01, nll_pos 3.414E-01, nll_neg 1.757E-01, num_matchable 1.157E+02, num_unmatchable 3.815E+02, confidence 2.060E-01, row_norm 9.520E-01}
[06/17/2024 08:17:30 gluefactory INFO] [E 9 | it 1000] loss {total 6.431E-01, last 1.554E-01, assignment_nll 1.554E-01, nll_pos 1.773E-01, nll_neg 1.334E-01, num_matchable 1.502E+02, num_unmatchable 3.448E+02, confidence 2.096E-01, row_norm 9.634E-01}
[06/17/2024 08:26:31 gluefactory INFO] [Validation] {match_recall 9.468E-01, match_precision 7.627E-01, accuracy 9.438E-01, average_precision 7.299E-01, loss/total 2.224E-01, loss/last 2.224E-01, loss/assignment_nll 2.224E-01, loss/nll_pos 2.984E-01, loss/nll_neg 1.463E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.554E-01}
[06/17/2024 08:29:06 gluefactory INFO] [E 9 | it 1100] loss {total 7.093E-01, last 2.007E-01, assignment_nll 2.007E-01, nll_pos 2.120E-01, nll_neg 1.895E-01, num_matchable 1.395E+02, num_unmatchable 3.577E+02, confidence 2.102E-01, row_norm 9.642E-01}
[06/17/2024 08:31:40 gluefactory INFO] [E 9 | it 1200] loss {total 7.392E-01, last 1.955E-01, assignment_nll 1.955E-01, nll_pos 2.190E-01, nll_neg 1.721E-01, num_matchable 1.363E+02, num_unmatchable 3.596E+02, confidence 2.104E-01, row_norm 9.534E-01}
[06/17/2024 08:34:14 gluefactory INFO] [E 9 | it 1300] loss {total 8.256E-01, last 2.407E-01, assignment_nll 2.407E-01, nll_pos 3.052E-01, nll_neg 1.763E-01, num_matchable 1.239E+02, num_unmatchable 3.729E+02, confidence 2.140E-01, row_norm 9.524E-01}
[06/17/2024 08:36:49 gluefactory INFO] [E 9 | it 1400] loss {total 7.476E-01, last 2.051E-01, assignment_nll 2.051E-01, nll_pos 2.657E-01, nll_neg 1.445E-01, num_matchable 1.187E+02, num_unmatchable 3.788E+02, confidence 2.210E-01, row_norm 9.582E-01}
[06/17/2024 08:39:23 gluefactory INFO] [E 9 | it 1500] loss {total 9.943E-01, last 3.604E-01, assignment_nll 3.604E-01, nll_pos 5.117E-01, nll_neg 2.091E-01, num_matchable 1.038E+02, num_unmatchable 3.941E+02, confidence 2.138E-01, row_norm 9.341E-01}
[06/17/2024 08:48:22 gluefactory INFO] [Validation] {match_recall 9.417E-01, match_precision 7.658E-01, accuracy 9.451E-01, average_precision 7.305E-01, loss/total 2.336E-01, loss/last 2.336E-01, loss/assignment_nll 2.336E-01, loss/nll_pos 3.132E-01, loss/nll_neg 1.540E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.606E-01}
[06/17/2024 08:50:58 gluefactory INFO] [E 9 | it 1600] loss {total 6.725E-01, last 1.516E-01, assignment_nll 1.516E-01, nll_pos 1.743E-01, nll_neg 1.288E-01, num_matchable 1.293E+02, num_unmatchable 3.673E+02, confidence 2.042E-01, row_norm 9.666E-01}
[06/17/2024 08:53:32 gluefactory INFO] [E 9 | it 1700] loss {total 7.693E-01, last 2.482E-01, assignment_nll 2.482E-01, nll_pos 3.557E-01, nll_neg 1.407E-01, num_matchable 1.459E+02, num_unmatchable 3.485E+02, confidence 2.146E-01, row_norm 9.483E-01}
[06/17/2024 08:56:06 gluefactory INFO] [E 9 | it 1800] loss {total 6.473E-01, last 1.362E-01, assignment_nll 1.362E-01, nll_pos 1.458E-01, nll_neg 1.267E-01, num_matchable 1.284E+02, num_unmatchable 3.684E+02, confidence 2.077E-01, row_norm 9.670E-01}
[06/17/2024 09:05:35 gluefactory INFO] [Validation] {match_recall 9.483E-01, match_precision 7.535E-01, accuracy 9.403E-01, average_precision 7.222E-01, loss/total 2.160E-01, loss/last 2.160E-01, loss/assignment_nll 2.160E-01, loss/nll_pos 2.547E-01, loss/nll_neg 1.772E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.552E-01}
[06/17/2024 09:05:35 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 09:05:37 gluefactory INFO] New best val: loss/total=0.21597073760518049
[06/17/2024 09:05:39 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_9_18239.tar
[06/17/2024 09:05:40 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_2_5000.tar
[06/17/2024 09:05:40 gluefactory INFO] Starting epoch 10
[06/17/2024 09:05:40 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/17/2024 09:05:48 gluefactory INFO] [E 10 | it 0] loss {total 7.535E-01, last 2.217E-01, assignment_nll 2.217E-01, nll_pos 2.702E-01, nll_neg 1.731E-01, num_matchable 1.352E+02, num_unmatchable 3.626E+02, confidence 2.077E-01, row_norm 9.622E-01}
[06/17/2024 09:08:23 gluefactory INFO] [E 10 | it 100] loss {total 8.160E-01, last 2.728E-01, assignment_nll 2.728E-01, nll_pos 3.873E-01, nll_neg 1.584E-01, num_matchable 1.199E+02, num_unmatchable 3.775E+02, confidence 2.115E-01, row_norm 9.559E-01}
[06/17/2024 09:10:57 gluefactory INFO] [E 10 | it 200] loss {total 8.764E-01, last 3.349E-01, assignment_nll 3.349E-01, nll_pos 4.844E-01, nll_neg 1.854E-01, num_matchable 1.357E+02, num_unmatchable 3.618E+02, confidence 2.119E-01, row_norm 9.405E-01}
[06/17/2024 09:13:30 gluefactory INFO] [E 10 | it 300] loss {total 6.523E-01, last 1.635E-01, assignment_nll 1.635E-01, nll_pos 2.072E-01, nll_neg 1.198E-01, num_matchable 1.322E+02, num_unmatchable 3.664E+02, confidence 1.942E-01, row_norm 9.667E-01}
[06/17/2024 09:16:05 gluefactory INFO] [E 10 | it 400] loss {total 7.411E-01, last 2.184E-01, assignment_nll 2.184E-01, nll_pos 2.850E-01, nll_neg 1.519E-01, num_matchable 1.163E+02, num_unmatchable 3.781E+02, confidence 2.084E-01, row_norm 9.654E-01}
[06/17/2024 09:18:42 gluefactory INFO] [E 10 | it 500] loss {total 8.799E-01, last 2.809E-01, assignment_nll 2.809E-01, nll_pos 3.443E-01, nll_neg 2.176E-01, num_matchable 1.198E+02, num_unmatchable 3.788E+02, confidence 2.294E-01, row_norm 9.371E-01}
[06/17/2024 09:27:33 gluefactory INFO] [Validation] {match_recall 9.328E-01, match_precision 7.354E-01, accuracy 9.335E-01, average_precision 6.985E-01, loss/total 2.842E-01, loss/last 2.842E-01, loss/assignment_nll 2.842E-01, loss/nll_pos 3.923E-01, loss/nll_neg 1.761E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.464E-01}
[06/17/2024 09:30:08 gluefactory INFO] [E 10 | it 600] loss {total 1.017E+00, last 4.096E-01, assignment_nll 4.096E-01, nll_pos 6.099E-01, nll_neg 2.094E-01, num_matchable 1.257E+02, num_unmatchable 3.720E+02, confidence 2.146E-01, row_norm 9.492E-01}
[06/17/2024 09:32:42 gluefactory INFO] [E 10 | it 700] loss {total 6.983E-01, last 1.974E-01, assignment_nll 1.974E-01, nll_pos 2.818E-01, nll_neg 1.130E-01, num_matchable 1.362E+02, num_unmatchable 3.625E+02, confidence 1.933E-01, row_norm 9.688E-01}
[06/17/2024 09:35:15 gluefactory INFO] [E 10 | it 800] loss {total 6.855E-01, last 1.569E-01, assignment_nll 1.569E-01, nll_pos 1.794E-01, nll_neg 1.344E-01, num_matchable 1.343E+02, num_unmatchable 3.605E+02, confidence 2.069E-01, row_norm 9.648E-01}
[06/17/2024 09:37:49 gluefactory INFO] [E 10 | it 900] loss {total 8.802E-01, last 3.502E-01, assignment_nll 3.502E-01, nll_pos 5.045E-01, nll_neg 1.960E-01, num_matchable 1.208E+02, num_unmatchable 3.755E+02, confidence 2.025E-01, row_norm 9.367E-01}
[06/17/2024 09:40:23 gluefactory INFO] [E 10 | it 1000] loss {total 6.792E-01, last 1.715E-01, assignment_nll 1.715E-01, nll_pos 2.013E-01, nll_neg 1.417E-01, num_matchable 1.472E+02, num_unmatchable 3.505E+02, confidence 2.077E-01, row_norm 9.598E-01}
[06/17/2024 09:49:13 gluefactory INFO] [Validation] {match_recall 9.438E-01, match_precision 7.673E-01, accuracy 9.455E-01, average_precision 7.338E-01, loss/total 2.247E-01, loss/last 2.247E-01, loss/assignment_nll 2.247E-01, loss/nll_pos 2.983E-01, loss/nll_neg 1.510E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.519E-01}
[06/17/2024 09:51:49 gluefactory INFO] [E 10 | it 1100] loss {total 7.065E-01, last 1.577E-01, assignment_nll 1.577E-01, nll_pos 1.767E-01, nll_neg 1.386E-01, num_matchable 1.357E+02, num_unmatchable 3.623E+02, confidence 2.100E-01, row_norm 9.689E-01}
[06/17/2024 09:54:23 gluefactory INFO] [E 10 | it 1200] loss {total 8.612E-01, last 3.449E-01, assignment_nll 3.449E-01, nll_pos 5.448E-01, nll_neg 1.451E-01, num_matchable 1.265E+02, num_unmatchable 3.700E+02, confidence 2.062E-01, row_norm 9.422E-01}
[06/17/2024 09:56:56 gluefactory INFO] [E 10 | it 1300] loss {total 7.607E-01, last 1.953E-01, assignment_nll 1.953E-01, nll_pos 2.308E-01, nll_neg 1.598E-01, num_matchable 1.275E+02, num_unmatchable 3.671E+02, confidence 2.122E-01, row_norm 9.556E-01}
[06/17/2024 09:59:30 gluefactory INFO] [E 10 | it 1400] loss {total 7.875E-01, last 2.013E-01, assignment_nll 2.013E-01, nll_pos 2.576E-01, nll_neg 1.450E-01, num_matchable 1.138E+02, num_unmatchable 3.841E+02, confidence 2.029E-01, row_norm 9.570E-01}
[06/17/2024 10:02:04 gluefactory INFO] [E 10 | it 1500] loss {total 8.217E-01, last 2.816E-01, assignment_nll 2.816E-01, nll_pos 4.025E-01, nll_neg 1.607E-01, num_matchable 1.216E+02, num_unmatchable 3.746E+02, confidence 2.042E-01, row_norm 9.552E-01}
[06/17/2024 10:10:55 gluefactory INFO] [Validation] {match_recall 9.511E-01, match_precision 7.770E-01, accuracy 9.493E-01, average_precision 7.453E-01, loss/total 1.966E-01, loss/last 1.966E-01, loss/assignment_nll 1.966E-01, loss/nll_pos 2.548E-01, loss/nll_neg 1.384E-01, loss/num_matchable 1.250E+02, loss/num_unmatchable 3.724E+02, loss/row_norm 9.648E-01}
[06/17/2024 10:10:55 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/17/2024 10:10:57 gluefactory INFO] New best val: loss/total=0.19657734884499098
[06/17/2024 10:13:32 gluefactory INFO] [E 10 | it 1600] loss {total 7.022E-01, last 1.925E-01, assignment_nll 1.925E-01, nll_pos 2.650E-01, nll_neg 1.200E-01, num_matchable 1.187E+02, num_unmatchable 3.793E+02, confidence 1.842E-01, row_norm 9.719E-01}
[06/17/2024 10:16:06 gluefactory INFO] [E 10 | it 1700] loss {total 7.963E-01, last 2.307E-01, assignment_nll 2.307E-01, nll_pos 3.320E-01, nll_neg 1.295E-01, num_matchable 1.408E+02, num_unmatchable 3.537E+02, confidence 2.162E-01, row_norm 9.389E-01}
[06/17/2024 10:17:39 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_10_20000.tar
[06/17/2024 10:17:40 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_2_5471.tar
[06/17/2024 10:18:42 gluefactory INFO] [E 10 | it 1800] loss {total 6.917E-01, last 1.802E-01, assignment_nll 1.802E-01, nll_pos 2.216E-01, nll_neg 1.388E-01, num_matchable 1.122E+02, num_unmatchable 3.854E+02, confidence 1.971E-01, row_norm 9.710E-01}


"""

# Example usage
ata = """
[06/13/2024 16:10:43 gluefactory INFO] [E 0 | it 0] loss {total 7.125E+00, last 6.127E+00, assignment_nll 6.127E+00, nll_pos 1.034E+01, nll_neg 1.916E+00, num_matchable 1.258E+02, num_unmatchable 3.737E+02, confidence 5.511E-01, row_norm 1.633E-01}
[06/13/2024 16:16:43 gluefactory INFO] [Validation] {match_recall 5.286E-05, match_precision 1.019E-02, accuracy 7.466E-01, average_precision 1.539E-06, loss/total 5.453E+00, loss/last 5.453E+00, loss/assignment_nll 5.453E+00, loss/nll_pos 1.008E+01, loss/nll_neg 8.260E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 4.566E-01}
[06/13/2024 16:16:43 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 16:16:44 gluefactory INFO] New best val: loss/total=5.453296089619772
[06/13/2024 16:20:09 gluefactory INFO] [E 0 | it 100] loss {total 3.115E+00, last 2.386E+00, assignment_nll 2.386E+00, nll_pos 3.777E+00, nll_neg 9.954E-01, num_matchable 1.269E+02, num_unmatchable 3.699E+02, confidence 4.241E-01, row_norm 4.872E-01}
[06/13/2024 16:23:34 gluefactory INFO] [E 0 | it 200] loss {total 2.866E+00, last 2.109E+00, assignment_nll 2.109E+00, nll_pos 3.352E+00, nll_neg 8.656E-01, num_matchable 1.301E+02, num_unmatchable 3.614E+02, confidence 3.927E-01, row_norm 5.821E-01}
[06/13/2024 16:26:56 gluefactory INFO] [E 0 | it 300] loss {total 2.557E+00, last 1.794E+00, assignment_nll 1.794E+00, nll_pos 2.954E+00, nll_neg 6.329E-01, num_matchable 1.169E+02, num_unmatchable 3.814E+02, confidence 3.398E-01, row_norm 6.820E-01}
[06/13/2024 16:30:19 gluefactory INFO] [E 0 | it 400] loss {total 2.270E+00, last 1.484E+00, assignment_nll 1.484E+00, nll_pos 2.244E+00, nll_neg 7.242E-01, num_matchable 1.161E+02, num_unmatchable 3.802E+02, confidence 3.609E-01, row_norm 6.772E-01}
[06/13/2024 16:33:41 gluefactory INFO] [E 0 | it 500] loss {total 2.056E+00, last 1.307E+00, assignment_nll 1.307E+00, nll_pos 1.927E+00, nll_neg 6.876E-01, num_matchable 1.268E+02, num_unmatchable 3.731E+02, confidence 3.437E-01, row_norm 6.970E-01}
[06/13/2024 16:39:36 gluefactory INFO] [Validation] {match_recall 7.081E-01, match_precision 5.632E-01, accuracy 8.451E-01, average_precision 4.320E-01, loss/total 1.228E+00, loss/last 1.228E+00, loss/assignment_nll 1.228E+00, loss/nll_pos 1.857E+00, loss/nll_neg 5.984E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 7.335E-01}
[06/13/2024 16:39:36 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 16:39:36 gluefactory INFO] New best val: loss/total=1.2275679911609068
[06/13/2024 16:43:02 gluefactory INFO] [E 0 | it 600] loss {total 1.825E+00, last 1.039E+00, assignment_nll 1.039E+00, nll_pos 1.547E+00, nll_neg 5.315E-01, num_matchable 1.197E+02, num_unmatchable 3.740E+02, confidence 3.185E-01, row_norm 7.736E-01}
[06/13/2024 16:46:27 gluefactory INFO] [E 0 | it 700] loss {total 1.835E+00, last 1.067E+00, assignment_nll 1.067E+00, nll_pos 1.559E+00, nll_neg 5.750E-01, num_matchable 1.261E+02, num_unmatchable 3.718E+02, confidence 3.284E-01, row_norm 7.567E-01}
[06/13/2024 16:49:51 gluefactory INFO] [E 0 | it 800] loss {total 1.733E+00, last 9.983E-01, assignment_nll 9.983E-01, nll_pos 1.478E+00, nll_neg 5.185E-01, num_matchable 1.242E+02, num_unmatchable 3.722E+02, confidence 3.058E-01, row_norm 7.881E-01}
[06/13/2024 16:53:15 gluefactory INFO] [E 0 | it 900] loss {total 1.734E+00, last 9.692E-01, assignment_nll 9.692E-01, nll_pos 1.358E+00, nll_neg 5.800E-01, num_matchable 1.292E+02, num_unmatchable 3.684E+02, confidence 3.211E-01, row_norm 7.721E-01}
[06/13/2024 16:56:39 gluefactory INFO] [E 0 | it 1000] loss {total 1.462E+00, last 7.428E-01, assignment_nll 7.428E-01, nll_pos 9.294E-01, nll_neg 5.562E-01, num_matchable 1.369E+02, num_unmatchable 3.603E+02, confidence 3.227E-01, row_norm 8.101E-01}
[06/13/2024 17:02:37 gluefactory INFO] [Validation] {match_recall 7.830E-01, match_precision 5.816E-01, accuracy 8.560E-01, average_precision 4.820E-01, loss/total 9.044E-01, loss/last 9.044E-01, loss/assignment_nll 9.044E-01, loss/nll_pos 1.247E+00, loss/nll_neg 5.614E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 7.882E-01}
[06/13/2024 17:02:37 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 17:02:38 gluefactory INFO] New best val: loss/total=0.9043711014552996
[06/13/2024 17:06:04 gluefactory INFO] [E 0 | it 1100] loss {total 1.437E+00, last 6.742E-01, assignment_nll 6.742E-01, nll_pos 9.109E-01, nll_neg 4.375E-01, num_matchable 1.370E+02, num_unmatchable 3.615E+02, confidence 3.166E-01, row_norm 8.481E-01}
[06/13/2024 17:09:28 gluefactory INFO] [E 0 | it 1200] loss {total 1.548E+00, last 8.023E-01, assignment_nll 8.023E-01, nll_pos 1.163E+00, nll_neg 4.412E-01, num_matchable 1.251E+02, num_unmatchable 3.695E+02, confidence 3.038E-01, row_norm 8.245E-01}
[06/13/2024 17:12:53 gluefactory INFO] [E 0 | it 1300] loss {total 1.522E+00, last 8.434E-01, assignment_nll 8.434E-01, nll_pos 1.213E+00, nll_neg 4.742E-01, num_matchable 1.258E+02, num_unmatchable 3.696E+02, confidence 2.806E-01, row_norm 8.119E-01}
[06/13/2024 17:16:17 gluefactory INFO] [E 0 | it 1400] loss {total 1.480E+00, last 8.306E-01, assignment_nll 8.306E-01, nll_pos 1.238E+00, nll_neg 4.226E-01, num_matchable 1.329E+02, num_unmatchable 3.661E+02, confidence 2.652E-01, row_norm 8.484E-01}
[06/13/2024 17:19:41 gluefactory INFO] [E 0 | it 1500] loss {total 1.450E+00, last 8.350E-01, assignment_nll 8.350E-01, nll_pos 1.226E+00, nll_neg 4.438E-01, num_matchable 1.339E+02, num_unmatchable 3.620E+02, confidence 2.688E-01, row_norm 8.320E-01}
[06/13/2024 17:25:40 gluefactory INFO] [Validation] {match_recall 8.238E-01, match_precision 6.144E-01, accuracy 8.722E-01, average_precision 5.293E-01, loss/total 7.143E-01, loss/last 7.143E-01, loss/assignment_nll 7.143E-01, loss/nll_pos 1.001E+00, loss/nll_neg 4.274E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 8.485E-01}
[06/13/2024 17:25:41 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 17:25:41 gluefactory INFO] New best val: loss/total=0.7142970442222294
[06/13/2024 17:33:11 gluefactory INFO] [Validation] {match_recall 8.231E-01, match_precision 6.090E-01, accuracy 8.696E-01, average_precision 5.244E-01, loss/total 7.234E-01, loss/last 7.234E-01, loss/assignment_nll 7.234E-01, loss/nll_pos 9.969E-01, loss/nll_neg 4.500E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 8.436E-01}
[06/13/2024 17:33:12 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_0_1544.tar
[06/13/2024 17:33:13 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_0_500_interrupted.tar
[06/13/2024 17:33:13 gluefactory INFO] Starting epoch 1
[06/13/2024 17:33:13 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/13/2024 17:33:18 gluefactory INFO] [E 1 | it 0] loss {total 1.427E+00, last 7.588E-01, assignment_nll 7.588E-01, nll_pos 1.046E+00, nll_neg 4.716E-01, num_matchable 1.238E+02, num_unmatchable 3.755E+02, confidence 2.767E-01, row_norm 8.476E-01}
[06/13/2024 17:36:59 gluefactory INFO] [E 1 | it 100] loss {total 1.236E+00, last 6.076E-01, assignment_nll 6.076E-01, nll_pos 7.868E-01, nll_neg 4.283E-01, num_matchable 1.296E+02, num_unmatchable 3.689E+02, confidence 2.675E-01, row_norm 8.535E-01}
[06/13/2024 17:40:23 gluefactory INFO] [E 1 | it 200] loss {total 1.333E+00, last 6.378E-01, assignment_nll 6.378E-01, nll_pos 9.128E-01, nll_neg 3.628E-01, num_matchable 1.341E+02, num_unmatchable 3.604E+02, confidence 2.625E-01, row_norm 8.446E-01}
[06/13/2024 17:43:48 gluefactory INFO] [E 1 | it 300] loss {total 1.426E+00, last 6.665E-01, assignment_nll 6.665E-01, nll_pos 9.245E-01, nll_neg 4.085E-01, num_matchable 1.258E+02, num_unmatchable 3.725E+02, confidence 2.902E-01, row_norm 8.636E-01}
[06/13/2024 17:47:12 gluefactory INFO] [E 1 | it 400] loss {total 1.434E+00, last 8.079E-01, assignment_nll 8.079E-01, nll_pos 1.166E+00, nll_neg 4.499E-01, num_matchable 1.160E+02, num_unmatchable 3.813E+02, confidence 2.617E-01, row_norm 7.935E-01}
[06/13/2024 17:50:36 gluefactory INFO] [E 1 | it 500] loss {total 1.304E+00, last 6.430E-01, assignment_nll 6.430E-01, nll_pos 8.407E-01, nll_neg 4.452E-01, num_matchable 1.154E+02, num_unmatchable 3.836E+02, confidence 2.709E-01, row_norm 8.342E-01}
[06/13/2024 17:56:35 gluefactory INFO] [Validation] {match_recall 8.465E-01, match_precision 6.148E-01, accuracy 8.719E-01, average_precision 5.411E-01, loss/total 6.390E-01, loss/last 6.390E-01, loss/assignment_nll 6.390E-01, loss/nll_pos 8.497E-01, loss/nll_neg 4.282E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 8.515E-01}
[06/13/2024 17:56:35 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 17:56:36 gluefactory INFO] New best val: loss/total=0.6389517079450296
[06/13/2024 18:00:02 gluefactory INFO] [E 1 | it 600] loss {total 1.165E+00, last 5.573E-01, assignment_nll 5.573E-01, nll_pos 7.665E-01, nll_neg 3.481E-01, num_matchable 1.240E+02, num_unmatchable 3.712E+02, confidence 2.599E-01, row_norm 8.853E-01}
[06/13/2024 18:03:27 gluefactory INFO] [E 1 | it 700] loss {total 1.157E+00, last 4.762E-01, assignment_nll 4.762E-01, nll_pos 5.501E-01, nll_neg 4.024E-01, num_matchable 1.219E+02, num_unmatchable 3.778E+02, confidence 2.663E-01, row_norm 8.941E-01}
[06/13/2024 18:06:52 gluefactory INFO] [E 1 | it 800] loss {total 1.350E+00, last 6.878E-01, assignment_nll 6.878E-01, nll_pos 9.370E-01, nll_neg 4.387E-01, num_matchable 1.288E+02, num_unmatchable 3.682E+02, confidence 2.732E-01, row_norm 8.414E-01}
[06/13/2024 18:10:17 gluefactory INFO] [E 1 | it 900] loss {total 1.183E+00, last 5.220E-01, assignment_nll 5.220E-01, nll_pos 6.290E-01, nll_neg 4.150E-01, num_matchable 1.262E+02, num_unmatchable 3.722E+02, confidence 2.682E-01, row_norm 8.778E-01}
[06/13/2024 18:13:42 gluefactory INFO] [E 1 | it 1000] loss {total 1.031E+00, last 4.516E-01, assignment_nll 4.516E-01, nll_pos 5.051E-01, nll_neg 3.982E-01, num_matchable 1.369E+02, num_unmatchable 3.615E+02, confidence 2.447E-01, row_norm 9.025E-01}
[06/13/2024 18:19:40 gluefactory INFO] [Validation] {match_recall 8.565E-01, match_precision 6.248E-01, accuracy 8.771E-01, average_precision 5.543E-01, loss/total 5.956E-01, loss/last 5.956E-01, loss/assignment_nll 5.956E-01, loss/nll_pos 7.461E-01, loss/nll_neg 4.452E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 8.801E-01}
[06/13/2024 18:19:40 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 18:19:41 gluefactory INFO] New best val: loss/total=0.5956461963593396
[06/13/2024 18:23:07 gluefactory INFO] [E 1 | it 1100] loss {total 1.236E+00, last 5.304E-01, assignment_nll 5.304E-01, nll_pos 6.539E-01, nll_neg 4.069E-01, num_matchable 1.223E+02, num_unmatchable 3.776E+02, confidence 2.689E-01, row_norm 8.785E-01}
[06/13/2024 18:26:30 gluefactory INFO] [E 1 | it 1200] loss {total 1.261E+00, last 6.184E-01, assignment_nll 6.184E-01, nll_pos 8.498E-01, nll_neg 3.871E-01, num_matchable 1.233E+02, num_unmatchable 3.733E+02, confidence 2.566E-01, row_norm 8.917E-01}
[06/13/2024 18:29:53 gluefactory INFO] [E 1 | it 1300] loss {total 1.246E+00, last 5.707E-01, assignment_nll 5.707E-01, nll_pos 7.400E-01, nll_neg 4.015E-01, num_matchable 1.294E+02, num_unmatchable 3.646E+02, confidence 2.751E-01, row_norm 8.669E-01}
[06/13/2024 18:33:16 gluefactory INFO] [E 1 | it 1400] loss {total 1.295E+00, last 6.568E-01, assignment_nll 6.568E-01, nll_pos 9.077E-01, nll_neg 4.059E-01, num_matchable 1.193E+02, num_unmatchable 3.773E+02, confidence 2.454E-01, row_norm 8.728E-01}
[06/13/2024 18:36:39 gluefactory INFO] [E 1 | it 1500] loss {total 1.032E+00, last 4.312E-01, assignment_nll 4.312E-01, nll_pos 4.380E-01, nll_neg 4.245E-01, num_matchable 1.369E+02, num_unmatchable 3.578E+02, confidence 2.732E-01, row_norm 8.853E-01}
[06/13/2024 18:42:35 gluefactory INFO] [Validation] {match_recall 8.670E-01, match_precision 6.283E-01, accuracy 8.790E-01, average_precision 5.627E-01, loss/total 5.532E-01, loss/last 5.532E-01, loss/assignment_nll 5.532E-01, loss/nll_pos 7.072E-01, loss/nll_neg 3.993E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 8.757E-01}
[06/13/2024 18:42:35 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 18:42:35 gluefactory INFO] New best val: loss/total=0.553216526290991
[06/13/2024 18:50:05 gluefactory INFO] [Validation] {match_recall 8.678E-01, match_precision 6.338E-01, accuracy 8.815E-01, average_precision 5.673E-01, loss/total 5.447E-01, loss/last 5.447E-01, loss/assignment_nll 5.447E-01, loss/nll_pos 6.918E-01, loss/nll_neg 3.976E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 8.859E-01}
[06/13/2024 18:50:05 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 18:50:06 gluefactory INFO] New best val: loss/total=0.5446704999341612
[06/13/2024 18:50:08 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_1_3089.tar
[06/13/2024 18:50:08 gluefactory INFO] Starting epoch 2
[06/13/2024 18:50:08 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/13/2024 18:50:14 gluefactory INFO] [E 2 | it 0] loss {total 1.225E+00, last 5.824E-01, assignment_nll 5.824E-01, nll_pos 7.610E-01, nll_neg 4.038E-01, num_matchable 1.203E+02, num_unmatchable 3.771E+02, confidence 2.487E-01, row_norm 8.878E-01}
[06/13/2024 18:53:38 gluefactory INFO] [E 2 | it 100] loss {total 1.153E+00, last 5.713E-01, assignment_nll 5.713E-01, nll_pos 7.342E-01, nll_neg 4.083E-01, num_matchable 1.353E+02, num_unmatchable 3.606E+02, confidence 2.444E-01, row_norm 8.752E-01}
[06/13/2024 18:57:03 gluefactory INFO] [E 2 | it 200] loss {total 1.054E+00, last 4.279E-01, assignment_nll 4.279E-01, nll_pos 5.525E-01, nll_neg 3.033E-01, num_matchable 1.445E+02, num_unmatchable 3.510E+02, confidence 2.520E-01, row_norm 9.002E-01}
[06/13/2024 19:00:28 gluefactory INFO] [E 2 | it 300] loss {total 1.153E+00, last 4.955E-01, assignment_nll 4.955E-01, nll_pos 6.409E-01, nll_neg 3.500E-01, num_matchable 1.213E+02, num_unmatchable 3.770E+02, confidence 2.507E-01, row_norm 8.779E-01}
[06/13/2024 19:03:53 gluefactory INFO] [E 2 | it 400] loss {total 1.374E+00, last 7.314E-01, assignment_nll 7.314E-01, nll_pos 1.091E+00, nll_neg 3.718E-01, num_matchable 1.124E+02, num_unmatchable 3.860E+02, confidence 2.378E-01, row_norm 8.571E-01}
[06/13/2024 19:07:18 gluefactory INFO] [E 2 | it 500] loss {total 1.085E+00, last 4.537E-01, assignment_nll 4.537E-01, nll_pos 5.245E-01, nll_neg 3.830E-01, num_matchable 1.375E+02, num_unmatchable 3.608E+02, confidence 2.614E-01, row_norm 8.985E-01}
[06/13/2024 19:13:15 gluefactory INFO] [Validation] {match_recall 8.654E-01, match_precision 6.341E-01, accuracy 8.825E-01, average_precision 5.672E-01, loss/total 5.566E-01, loss/last 5.566E-01, loss/assignment_nll 5.566E-01, loss/nll_pos 7.386E-01, loss/nll_neg 3.746E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 8.875E-01}
[06/13/2024 19:16:42 gluefactory INFO] [E 2 | it 600] loss {total 9.648E-01, last 3.742E-01, assignment_nll 3.742E-01, nll_pos 4.403E-01, nll_neg 3.081E-01, num_matchable 1.234E+02, num_unmatchable 3.726E+02, confidence 2.340E-01, row_norm 9.192E-01}
[06/13/2024 19:20:07 gluefactory INFO] [E 2 | it 700] loss {total 1.068E+00, last 4.702E-01, assignment_nll 4.702E-01, nll_pos 6.119E-01, nll_neg 3.286E-01, num_matchable 1.324E+02, num_unmatchable 3.672E+02, confidence 2.296E-01, row_norm 9.102E-01}
[06/13/2024 19:23:32 gluefactory INFO] [E 2 | it 800] loss {total 1.023E+00, last 4.265E-01, assignment_nll 4.265E-01, nll_pos 4.746E-01, nll_neg 3.784E-01, num_matchable 1.257E+02, num_unmatchable 3.702E+02, confidence 2.417E-01, row_norm 9.065E-01}
[06/13/2024 19:26:57 gluefactory INFO] [E 2 | it 900] loss {total 9.613E-01, last 3.674E-01, assignment_nll 3.674E-01, nll_pos 3.824E-01, nll_neg 3.524E-01, num_matchable 1.274E+02, num_unmatchable 3.724E+02, confidence 2.358E-01, row_norm 9.209E-01}
[06/13/2024 19:30:22 gluefactory INFO] [E 2 | it 1000] loss {total 1.097E+00, last 5.048E-01, assignment_nll 5.048E-01, nll_pos 6.271E-01, nll_neg 3.824E-01, num_matchable 1.458E+02, num_unmatchable 3.491E+02, confidence 2.427E-01, row_norm 9.102E-01}
[06/13/2024 19:36:20 gluefactory INFO] [Validation] {match_recall 8.723E-01, match_precision 6.386E-01, accuracy 8.846E-01, average_precision 5.752E-01, loss/total 5.377E-01, loss/last 5.377E-01, loss/assignment_nll 5.377E-01, loss/nll_pos 6.994E-01, loss/nll_neg 3.760E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.001E-01}
[06/13/2024 19:36:20 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 19:36:21 gluefactory INFO] New best val: loss/total=0.5376605929694284
[06/13/2024 19:39:48 gluefactory INFO] [E 2 | it 1100] loss {total 9.849E-01, last 4.001E-01, assignment_nll 4.001E-01, nll_pos 4.912E-01, nll_neg 3.090E-01, num_matchable 1.340E+02, num_unmatchable 3.648E+02, confidence 2.312E-01, row_norm 9.154E-01}
[06/13/2024 19:43:13 gluefactory INFO] [E 2 | it 1200] loss {total 1.033E+00, last 4.758E-01, assignment_nll 4.758E-01, nll_pos 6.312E-01, nll_neg 3.204E-01, num_matchable 1.295E+02, num_unmatchable 3.680E+02, confidence 2.206E-01, row_norm 9.074E-01}
[06/13/2024 19:46:38 gluefactory INFO] [E 2 | it 1300] loss {total 9.876E-01, last 4.112E-01, assignment_nll 4.112E-01, nll_pos 4.879E-01, nll_neg 3.344E-01, num_matchable 1.288E+02, num_unmatchable 3.676E+02, confidence 2.357E-01, row_norm 9.154E-01}
[06/13/2024 19:50:03 gluefactory INFO] [E 2 | it 1400] loss {total 1.141E+00, last 5.754E-01, assignment_nll 5.754E-01, nll_pos 8.394E-01, nll_neg 3.115E-01, num_matchable 1.352E+02, num_unmatchable 3.628E+02, confidence 2.291E-01, row_norm 9.032E-01}
[06/13/2024 19:53:28 gluefactory INFO] [E 2 | it 1500] loss {total 1.004E+00, last 4.481E-01, assignment_nll 4.481E-01, nll_pos 5.691E-01, nll_neg 3.272E-01, num_matchable 1.408E+02, num_unmatchable 3.552E+02, confidence 2.231E-01, row_norm 9.176E-01}
[06/13/2024 19:59:26 gluefactory INFO] [Validation] {match_recall 8.762E-01, match_precision 6.416E-01, accuracy 8.862E-01, average_precision 5.794E-01, loss/total 5.183E-01, loss/last 5.183E-01, loss/assignment_nll 5.183E-01, loss/nll_pos 6.330E-01, loss/nll_neg 4.035E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.000E-01}
[06/13/2024 19:59:26 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 19:59:27 gluefactory INFO] New best val: loss/total=0.5182529170372566
[06/13/2024 20:06:58 gluefactory INFO] [Validation] {match_recall 8.835E-01, match_precision 6.418E-01, accuracy 8.855E-01, average_precision 5.828E-01, loss/total 4.915E-01, loss/last 4.915E-01, loss/assignment_nll 4.915E-01, loss/nll_pos 6.092E-01, loss/nll_neg 3.738E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 8.987E-01}
[06/13/2024 20:06:58 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 20:06:58 gluefactory INFO] New best val: loss/total=0.49150731244789575
[06/13/2024 20:07:00 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_2_4634.tar
[06/13/2024 20:07:01 gluefactory INFO] Starting epoch 3
[06/13/2024 20:07:01 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/13/2024 20:07:06 gluefactory INFO] [E 3 | it 0] loss {total 1.055E+00, last 4.714E-01, assignment_nll 4.714E-01, nll_pos 5.571E-01, nll_neg 3.857E-01, num_matchable 1.272E+02, num_unmatchable 3.721E+02, confidence 2.318E-01, row_norm 9.209E-01}
[06/13/2024 20:10:31 gluefactory INFO] [E 3 | it 100] loss {total 9.821E-01, last 4.228E-01, assignment_nll 4.228E-01, nll_pos 5.011E-01, nll_neg 3.445E-01, num_matchable 1.272E+02, num_unmatchable 3.708E+02, confidence 2.213E-01, row_norm 9.082E-01}
[06/13/2024 20:13:56 gluefactory INFO] [E 3 | it 200] loss {total 9.883E-01, last 4.144E-01, assignment_nll 4.144E-01, nll_pos 5.436E-01, nll_neg 2.852E-01, num_matchable 1.260E+02, num_unmatchable 3.690E+02, confidence 2.209E-01, row_norm 9.100E-01}
[06/13/2024 20:17:21 gluefactory INFO] [E 3 | it 300] loss {total 1.284E+00, last 6.621E-01, assignment_nll 6.621E-01, nll_pos 9.962E-01, nll_neg 3.280E-01, num_matchable 1.163E+02, num_unmatchable 3.799E+02, confidence 2.249E-01, row_norm 8.829E-01}
[06/13/2024 20:19:34 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_3_5000.tar
[06/13/2024 20:20:47 gluefactory INFO] [E 3 | it 400] loss {total 1.159E+00, last 5.310E-01, assignment_nll 5.310E-01, nll_pos 7.108E-01, nll_neg 3.512E-01, num_matchable 1.178E+02, num_unmatchable 3.803E+02, confidence 2.337E-01, row_norm 8.963E-01}
[06/13/2024 20:24:12 gluefactory INFO] [E 3 | it 500] loss {total 9.394E-01, last 3.576E-01, assignment_nll 3.576E-01, nll_pos 3.718E-01, nll_neg 3.434E-01, num_matchable 1.278E+02, num_unmatchable 3.705E+02, confidence 2.434E-01, row_norm 9.170E-01}
[06/13/2024 20:30:07 gluefactory INFO] [Validation] {match_recall 8.924E-01, match_precision 6.489E-01, accuracy 8.893E-01, average_precision 5.937E-01, loss/total 4.555E-01, loss/last 4.555E-01, loss/assignment_nll 4.555E-01, loss/nll_pos 5.717E-01, loss/nll_neg 3.392E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.075E-01}
[06/13/2024 20:30:08 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 20:30:08 gluefactory INFO] New best val: loss/total=0.4554508008357536
[06/13/2024 20:33:33 gluefactory INFO] [E 3 | it 600] loss {total 1.002E+00, last 4.579E-01, assignment_nll 4.579E-01, nll_pos 5.793E-01, nll_neg 3.365E-01, num_matchable 1.207E+02, num_unmatchable 3.758E+02, confidence 2.253E-01, row_norm 9.140E-01}
[06/13/2024 20:36:56 gluefactory INFO] [E 3 | it 700] loss {total 9.504E-01, last 3.769E-01, assignment_nll 3.769E-01, nll_pos 4.354E-01, nll_neg 3.183E-01, num_matchable 1.325E+02, num_unmatchable 3.651E+02, confidence 2.279E-01, row_norm 9.213E-01}
[06/13/2024 20:40:19 gluefactory INFO] [E 3 | it 800] loss {total 1.076E+00, last 5.000E-01, assignment_nll 5.000E-01, nll_pos 6.839E-01, nll_neg 3.161E-01, num_matchable 1.108E+02, num_unmatchable 3.858E+02, confidence 2.229E-01, row_norm 9.128E-01}
[06/13/2024 20:43:44 gluefactory INFO] [E 3 | it 900] loss {total 1.110E+00, last 5.467E-01, assignment_nll 5.467E-01, nll_pos 7.268E-01, nll_neg 3.665E-01, num_matchable 1.208E+02, num_unmatchable 3.798E+02, confidence 2.239E-01, row_norm 8.952E-01}
[06/13/2024 20:47:09 gluefactory INFO] [E 3 | it 1000] loss {total 8.237E-01, last 3.329E-01, assignment_nll 3.329E-01, nll_pos 3.609E-01, nll_neg 3.049E-01, num_matchable 1.531E+02, num_unmatchable 3.448E+02, confidence 2.087E-01, row_norm 9.388E-01}
[06/13/2024 20:53:07 gluefactory INFO] [Validation] {match_recall 8.972E-01, match_precision 6.620E-01, accuracy 8.959E-01, average_precision 6.081E-01, loss/total 4.344E-01, loss/last 4.344E-01, loss/assignment_nll 4.344E-01, loss/nll_pos 5.470E-01, loss/nll_neg 3.219E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.179E-01}
[06/13/2024 20:53:07 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 20:53:08 gluefactory INFO] New best val: loss/total=0.4344478976829706
[06/13/2024 20:56:35 gluefactory INFO] [E 3 | it 1100] loss {total 9.336E-01, last 3.521E-01, assignment_nll 3.521E-01, nll_pos 3.861E-01, nll_neg 3.181E-01, num_matchable 1.322E+02, num_unmatchable 3.655E+02, confidence 2.325E-01, row_norm 9.190E-01}
[06/13/2024 21:00:00 gluefactory INFO] [E 3 | it 1200] loss {total 8.699E-01, last 3.440E-01, assignment_nll 3.440E-01, nll_pos 3.889E-01, nll_neg 2.991E-01, num_matchable 1.255E+02, num_unmatchable 3.710E+02, confidence 2.182E-01, row_norm 9.309E-01}
[06/13/2024 21:03:25 gluefactory INFO] [E 3 | it 1300] loss {total 8.677E-01, last 3.268E-01, assignment_nll 3.268E-01, nll_pos 3.555E-01, nll_neg 2.981E-01, num_matchable 1.283E+02, num_unmatchable 3.687E+02, confidence 2.235E-01, row_norm 9.353E-01}
[06/13/2024 21:06:51 gluefactory INFO] [E 3 | it 1400] loss {total 1.045E+00, last 5.157E-01, assignment_nll 5.157E-01, nll_pos 7.472E-01, nll_neg 2.842E-01, num_matchable 1.444E+02, num_unmatchable 3.539E+02, confidence 2.230E-01, row_norm 9.110E-01}
[06/13/2024 21:10:16 gluefactory INFO] [E 3 | it 1500] loss {total 1.038E+00, last 5.204E-01, assignment_nll 5.204E-01, nll_pos 7.578E-01, nll_neg 2.829E-01, num_matchable 1.398E+02, num_unmatchable 3.558E+02, confidence 2.150E-01, row_norm 9.181E-01}
[06/13/2024 21:16:15 gluefactory INFO] [Validation] {match_recall 8.971E-01, match_precision 6.743E-01, accuracy 9.025E-01, average_precision 6.189E-01, loss/total 4.177E-01, loss/last 4.177E-01, loss/assignment_nll 4.177E-01, loss/nll_pos 5.305E-01, loss/nll_neg 3.049E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.196E-01}
[06/13/2024 21:16:15 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 21:16:16 gluefactory INFO] New best val: loss/total=0.4176925691423954
[06/13/2024 21:23:47 gluefactory INFO] [Validation] {match_recall 8.980E-01, match_precision 6.744E-01, accuracy 9.022E-01, average_precision 6.190E-01, loss/total 4.178E-01, loss/last 4.178E-01, loss/assignment_nll 4.178E-01, loss/nll_pos 5.441E-01, loss/nll_neg 2.915E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.124E-01}
[06/13/2024 21:23:49 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_3_6179.tar
[06/13/2024 21:23:49 gluefactory INFO] Starting epoch 4
[06/13/2024 21:23:49 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/13/2024 21:23:55 gluefactory INFO] [E 4 | it 0] loss {total 9.650E-01, last 4.258E-01, assignment_nll 4.258E-01, nll_pos 5.692E-01, nll_neg 2.825E-01, num_matchable 1.252E+02, num_unmatchable 3.737E+02, confidence 2.224E-01, row_norm 9.194E-01}
[06/13/2024 21:27:20 gluefactory INFO] [E 4 | it 100] loss {total 1.113E+00, last 5.529E-01, assignment_nll 5.529E-01, nll_pos 7.620E-01, nll_neg 3.439E-01, num_matchable 1.274E+02, num_unmatchable 3.664E+02, confidence 2.208E-01, row_norm 8.980E-01}
[06/13/2024 21:30:45 gluefactory INFO] [E 4 | it 200] loss {total 1.038E+00, last 5.193E-01, assignment_nll 5.193E-01, nll_pos 7.430E-01, nll_neg 2.957E-01, num_matchable 1.294E+02, num_unmatchable 3.620E+02, confidence 2.384E-01, row_norm 9.195E-01}
[06/13/2024 21:34:11 gluefactory INFO] [E 4 | it 300] loss {total 1.216E+00, last 6.108E-01, assignment_nll 6.108E-01, nll_pos 8.942E-01, nll_neg 3.275E-01, num_matchable 1.159E+02, num_unmatchable 3.831E+02, confidence 2.192E-01, row_norm 8.876E-01}
[06/13/2024 21:37:36 gluefactory INFO] [E 4 | it 400] loss {total 1.025E+00, last 4.140E-01, assignment_nll 4.140E-01, nll_pos 4.948E-01, nll_neg 3.332E-01, num_matchable 1.223E+02, num_unmatchable 3.753E+02, confidence 2.481E-01, row_norm 8.893E-01}
[06/13/2024 21:41:01 gluefactory INFO] [E 4 | it 500] loss {total 9.504E-01, last 3.921E-01, assignment_nll 3.921E-01, nll_pos 4.362E-01, nll_neg 3.481E-01, num_matchable 1.312E+02, num_unmatchable 3.664E+02, confidence 2.417E-01, row_norm 9.007E-01}
[06/13/2024 21:47:00 gluefactory INFO] [Validation] {match_recall 8.900E-01, match_precision 6.702E-01, accuracy 9.008E-01, average_precision 6.128E-01, loss/total 4.481E-01, loss/last 4.481E-01, loss/assignment_nll 4.481E-01, loss/nll_pos 5.925E-01, loss/nll_neg 3.038E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.023E-01}
[06/13/2024 21:50:27 gluefactory INFO] [E 4 | it 600] loss {total 9.020E-01, last 3.372E-01, assignment_nll 3.372E-01, nll_pos 3.941E-01, nll_neg 2.802E-01, num_matchable 1.227E+02, num_unmatchable 3.726E+02, confidence 2.276E-01, row_norm 9.321E-01}
[06/13/2024 21:53:52 gluefactory INFO] [E 4 | it 700] loss {total 8.887E-01, last 3.382E-01, assignment_nll 3.382E-01, nll_pos 4.003E-01, nll_neg 2.761E-01, num_matchable 1.182E+02, num_unmatchable 3.811E+02, confidence 2.103E-01, row_norm 9.253E-01}
[06/13/2024 21:57:17 gluefactory INFO] [E 4 | it 800] loss {total 9.561E-01, last 3.717E-01, assignment_nll 3.717E-01, nll_pos 4.851E-01, nll_neg 2.584E-01, num_matchable 1.219E+02, num_unmatchable 3.744E+02, confidence 2.293E-01, row_norm 9.219E-01}
[06/13/2024 22:00:42 gluefactory INFO] [E 4 | it 900] loss {total 1.051E+00, last 4.802E-01, assignment_nll 4.802E-01, nll_pos 6.168E-01, nll_neg 3.436E-01, num_matchable 1.192E+02, num_unmatchable 3.780E+02, confidence 2.325E-01, row_norm 9.064E-01}
[06/13/2024 22:04:08 gluefactory INFO] [E 4 | it 1000] loss {total 8.430E-01, last 3.501E-01, assignment_nll 3.501E-01, nll_pos 4.165E-01, nll_neg 2.837E-01, num_matchable 1.419E+02, num_unmatchable 3.516E+02, confidence 2.094E-01, row_norm 9.392E-01}
[06/13/2024 22:10:07 gluefactory INFO] [Validation] {match_recall 9.074E-01, match_precision 6.847E-01, accuracy 9.074E-01, average_precision 6.339E-01, loss/total 3.899E-01, loss/last 3.899E-01, loss/assignment_nll 3.899E-01, loss/nll_pos 4.792E-01, loss/nll_neg 3.006E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.311E-01}
[06/13/2024 22:10:07 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 22:10:07 gluefactory INFO] New best val: loss/total=0.3899336093395628
[06/13/2024 22:13:34 gluefactory INFO] [E 4 | it 1100] loss {total 7.501E-01, last 2.421E-01, assignment_nll 2.421E-01, nll_pos 2.494E-01, nll_neg 2.348E-01, num_matchable 1.317E+02, num_unmatchable 3.676E+02, confidence 2.104E-01, row_norm 9.505E-01}
[06/13/2024 22:17:00 gluefactory INFO] [E 4 | it 1200] loss {total 8.558E-01, last 3.092E-01, assignment_nll 3.092E-01, nll_pos 3.701E-01, nll_neg 2.482E-01, num_matchable 1.304E+02, num_unmatchable 3.656E+02, confidence 2.247E-01, row_norm 9.354E-01}
[06/13/2024 22:20:25 gluefactory INFO] [E 4 | it 1300] loss {total 1.189E+00, last 6.072E-01, assignment_nll 6.072E-01, nll_pos 9.488E-01, nll_neg 2.655E-01, num_matchable 1.304E+02, num_unmatchable 3.657E+02, confidence 2.249E-01, row_norm 9.035E-01}
[06/13/2024 22:23:50 gluefactory INFO] [E 4 | it 1400] loss {total 8.675E-01, last 3.420E-01, assignment_nll 3.420E-01, nll_pos 4.545E-01, nll_neg 2.296E-01, num_matchable 1.412E+02, num_unmatchable 3.580E+02, confidence 2.065E-01, row_norm 9.430E-01}
[06/13/2024 22:27:15 gluefactory INFO] [E 4 | it 1500] loss {total 8.608E-01, last 3.591E-01, assignment_nll 3.591E-01, nll_pos 4.740E-01, nll_neg 2.441E-01, num_matchable 1.404E+02, num_unmatchable 3.533E+02, confidence 2.138E-01, row_norm 9.365E-01}
[06/13/2024 22:33:11 gluefactory INFO] [Validation] {match_recall 8.981E-01, match_precision 6.871E-01, accuracy 9.092E-01, average_precision 6.326E-01, loss/total 4.094E-01, loss/last 4.094E-01, loss/assignment_nll 4.094E-01, loss/nll_pos 5.395E-01, loss/nll_neg 2.793E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.241E-01}
[06/13/2024 22:40:37 gluefactory INFO] [Validation] {match_recall 9.054E-01, match_precision 6.890E-01, accuracy 9.096E-01, average_precision 6.372E-01, loss/total 3.857E-01, loss/last 3.857E-01, loss/assignment_nll 3.857E-01, loss/nll_pos 4.826E-01, loss/nll_neg 2.888E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.325E-01}
[06/13/2024 22:40:37 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 22:40:38 gluefactory INFO] New best val: loss/total=0.38570205954916026
[06/13/2024 22:40:40 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_4_7724.tar
[06/13/2024 22:40:40 gluefactory INFO] Starting epoch 5
[06/13/2024 22:40:40 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/13/2024 22:40:46 gluefactory INFO] [E 5 | it 0] loss {total 1.010E+00, last 4.686E-01, assignment_nll 4.686E-01, nll_pos 6.214E-01, nll_neg 3.157E-01, num_matchable 1.339E+02, num_unmatchable 3.641E+02, confidence 2.257E-01, row_norm 9.257E-01}
[06/13/2024 22:44:11 gluefactory INFO] [E 5 | it 100] loss {total 8.629E-01, last 3.366E-01, assignment_nll 3.366E-01, nll_pos 4.174E-01, nll_neg 2.557E-01, num_matchable 1.301E+02, num_unmatchable 3.671E+02, confidence 2.137E-01, row_norm 9.403E-01}
[06/13/2024 22:47:36 gluefactory INFO] [E 5 | it 200] loss {total 8.932E-01, last 3.415E-01, assignment_nll 3.415E-01, nll_pos 4.408E-01, nll_neg 2.421E-01, num_matchable 1.181E+02, num_unmatchable 3.782E+02, confidence 2.008E-01, row_norm 9.353E-01}
[06/13/2024 22:51:01 gluefactory INFO] [E 5 | it 300] loss {total 8.883E-01, last 3.105E-01, assignment_nll 3.105E-01, nll_pos 3.755E-01, nll_neg 2.454E-01, num_matchable 1.174E+02, num_unmatchable 3.803E+02, confidence 2.226E-01, row_norm 9.318E-01}
[06/13/2024 22:54:27 gluefactory INFO] [E 5 | it 400] loss {total 9.918E-01, last 4.506E-01, assignment_nll 4.506E-01, nll_pos 6.576E-01, nll_neg 2.437E-01, num_matchable 1.191E+02, num_unmatchable 3.787E+02, confidence 2.047E-01, row_norm 9.304E-01}
[06/13/2024 22:57:52 gluefactory INFO] [E 5 | it 500] loss {total 7.912E-01, last 2.577E-01, assignment_nll 2.577E-01, nll_pos 2.615E-01, nll_neg 2.539E-01, num_matchable 1.304E+02, num_unmatchable 3.689E+02, confidence 2.204E-01, row_norm 9.433E-01}
[06/13/2024 23:03:52 gluefactory INFO] [Validation] {match_recall 9.124E-01, match_precision 6.929E-01, accuracy 9.120E-01, average_precision 6.449E-01, loss/total 3.589E-01, loss/last 3.589E-01, loss/assignment_nll 3.589E-01, loss/nll_pos 4.720E-01, loss/nll_neg 2.458E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.252E-01}
[06/13/2024 23:03:52 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 23:03:53 gluefactory INFO] New best val: loss/total=0.3588791177041957
[06/13/2024 23:07:20 gluefactory INFO] [E 5 | it 600] loss {total 7.675E-01, last 2.600E-01, assignment_nll 2.600E-01, nll_pos 2.791E-01, nll_neg 2.410E-01, num_matchable 1.228E+02, num_unmatchable 3.736E+02, confidence 2.136E-01, row_norm 9.449E-01}
[06/13/2024 23:10:45 gluefactory INFO] [E 5 | it 700] loss {total 8.280E-01, last 2.710E-01, assignment_nll 2.710E-01, nll_pos 3.094E-01, nll_neg 2.325E-01, num_matchable 1.295E+02, num_unmatchable 3.694E+02, confidence 2.194E-01, row_norm 9.405E-01}
[06/13/2024 23:14:11 gluefactory INFO] [E 5 | it 800] loss {total 1.048E+00, last 4.444E-01, assignment_nll 4.444E-01, nll_pos 6.559E-01, nll_neg 2.328E-01, num_matchable 1.226E+02, num_unmatchable 3.711E+02, confidence 2.324E-01, row_norm 9.268E-01}
[06/13/2024 23:17:36 gluefactory INFO] [E 5 | it 900] loss {total 8.995E-01, last 3.597E-01, assignment_nll 3.597E-01, nll_pos 4.625E-01, nll_neg 2.569E-01, num_matchable 1.310E+02, num_unmatchable 3.662E+02, confidence 2.118E-01, row_norm 9.342E-01}
[06/13/2024 23:21:02 gluefactory INFO] [E 5 | it 1000] loss {total 8.848E-01, last 3.311E-01, assignment_nll 3.311E-01, nll_pos 3.810E-01, nll_neg 2.813E-01, num_matchable 1.375E+02, num_unmatchable 3.596E+02, confidence 2.167E-01, row_norm 9.522E-01}
[06/13/2024 23:27:01 gluefactory INFO] [Validation] {match_recall 9.072E-01, match_precision 6.898E-01, accuracy 9.103E-01, average_precision 6.389E-01, loss/total 3.927E-01, loss/last 3.927E-01, loss/assignment_nll 3.927E-01, loss/nll_pos 4.845E-01, loss/nll_neg 3.010E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.398E-01}
[06/13/2024 23:30:28 gluefactory INFO] [E 5 | it 1100] loss {total 8.048E-01, last 3.038E-01, assignment_nll 3.038E-01, nll_pos 3.908E-01, nll_neg 2.169E-01, num_matchable 1.342E+02, num_unmatchable 3.649E+02, confidence 2.070E-01, row_norm 9.410E-01}
[06/13/2024 23:33:53 gluefactory INFO] [E 5 | it 1200] loss {total 8.444E-01, last 2.736E-01, assignment_nll 2.736E-01, nll_pos 3.216E-01, nll_neg 2.256E-01, num_matchable 1.339E+02, num_unmatchable 3.640E+02, confidence 2.176E-01, row_norm 9.505E-01}
[06/13/2024 23:37:18 gluefactory INFO] [E 5 | it 1300] loss {total 8.524E-01, last 3.061E-01, assignment_nll 3.061E-01, nll_pos 3.826E-01, nll_neg 2.296E-01, num_matchable 1.262E+02, num_unmatchable 3.691E+02, confidence 2.174E-01, row_norm 9.321E-01}
[06/13/2024 23:40:44 gluefactory INFO] [E 5 | it 1400] loss {total 8.731E-01, last 3.441E-01, assignment_nll 3.441E-01, nll_pos 4.548E-01, nll_neg 2.335E-01, num_matchable 1.357E+02, num_unmatchable 3.627E+02, confidence 2.130E-01, row_norm 9.383E-01}
[06/13/2024 23:44:10 gluefactory INFO] [E 5 | it 1500] loss {total 7.089E-01, last 2.346E-01, assignment_nll 2.346E-01, nll_pos 2.514E-01, nll_neg 2.178E-01, num_matchable 1.465E+02, num_unmatchable 3.502E+02, confidence 2.019E-01, row_norm 9.489E-01}
[06/13/2024 23:50:09 gluefactory INFO] [Validation] {match_recall 9.220E-01, match_precision 7.088E-01, accuracy 9.194E-01, average_precision 6.645E-01, loss/total 3.248E-01, loss/last 3.248E-01, loss/assignment_nll 3.248E-01, loss/nll_pos 4.193E-01, loss/nll_neg 2.304E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.429E-01}
[06/13/2024 23:50:09 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/13/2024 23:50:10 gluefactory INFO] New best val: loss/total=0.3248223981492431
[06/13/2024 23:57:42 gluefactory INFO] [Validation] {match_recall 9.164E-01, match_precision 6.939E-01, accuracy 9.125E-01, average_precision 6.486E-01, loss/total 3.478E-01, loss/last 3.478E-01, loss/assignment_nll 3.478E-01, loss/nll_pos 4.270E-01, loss/nll_neg 2.686E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.302E-01}
[06/13/2024 23:57:44 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_5_9269.tar
[06/13/2024 23:57:45 gluefactory INFO] Starting epoch 6
[06/13/2024 23:57:45 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/13/2024 23:57:50 gluefactory INFO] [E 6 | it 0] loss {total 9.281E-01, last 3.697E-01, assignment_nll 3.697E-01, nll_pos 4.613E-01, nll_neg 2.781E-01, num_matchable 1.271E+02, num_unmatchable 3.708E+02, confidence 2.266E-01, row_norm 9.401E-01}
[06/14/2024 00:01:16 gluefactory INFO] [E 6 | it 100] loss {total 8.391E-01, last 3.521E-01, assignment_nll 3.521E-01, nll_pos 4.688E-01, nll_neg 2.354E-01, num_matchable 1.357E+02, num_unmatchable 3.617E+02, confidence 2.043E-01, row_norm 9.325E-01}
[06/14/2024 00:04:42 gluefactory INFO] [E 6 | it 200] loss {total 7.358E-01, last 2.188E-01, assignment_nll 2.188E-01, nll_pos 2.466E-01, nll_neg 1.910E-01, num_matchable 1.284E+02, num_unmatchable 3.672E+02, confidence 2.029E-01, row_norm 9.487E-01}
[06/14/2024 00:08:07 gluefactory INFO] [E 6 | it 300] loss {total 8.721E-01, last 2.725E-01, assignment_nll 2.725E-01, nll_pos 3.056E-01, nll_neg 2.394E-01, num_matchable 1.252E+02, num_unmatchable 3.721E+02, confidence 2.294E-01, row_norm 9.480E-01}
[06/14/2024 00:11:33 gluefactory INFO] [E 6 | it 400] loss {total 8.590E-01, last 3.075E-01, assignment_nll 3.075E-01, nll_pos 3.924E-01, nll_neg 2.226E-01, num_matchable 1.141E+02, num_unmatchable 3.834E+02, confidence 2.086E-01, row_norm 9.422E-01}
[06/14/2024 00:14:58 gluefactory INFO] [E 6 | it 500] loss {total 7.986E-01, last 2.352E-01, assignment_nll 2.352E-01, nll_pos 2.142E-01, nll_neg 2.562E-01, num_matchable 1.369E+02, num_unmatchable 3.615E+02, confidence 2.251E-01, row_norm 9.432E-01}
[06/14/2024 00:20:58 gluefactory INFO] [Validation] {match_recall 9.171E-01, match_precision 7.008E-01, accuracy 9.158E-01, average_precision 6.557E-01, loss/total 3.408E-01, loss/last 3.408E-01, loss/assignment_nll 3.408E-01, loss/nll_pos 4.289E-01, loss/nll_neg 2.527E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.328E-01}
[06/14/2024 00:24:25 gluefactory INFO] [E 6 | it 600] loss {total 7.965E-01, last 2.374E-01, assignment_nll 2.374E-01, nll_pos 2.671E-01, nll_neg 2.077E-01, num_matchable 1.305E+02, num_unmatchable 3.654E+02, confidence 2.202E-01, row_norm 9.542E-01}
[06/14/2024 00:27:50 gluefactory INFO] [E 6 | it 700] loss {total 8.242E-01, last 2.793E-01, assignment_nll 2.793E-01, nll_pos 3.060E-01, nll_neg 2.526E-01, num_matchable 1.242E+02, num_unmatchable 3.751E+02, confidence 2.111E-01, row_norm 9.467E-01}
[06/14/2024 00:28:51 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_6_10000.tar
[06/14/2024 00:31:14 gluefactory INFO] [E 6 | it 800] loss {total 8.831E-01, last 3.546E-01, assignment_nll 3.546E-01, nll_pos 4.566E-01, nll_neg 2.527E-01, num_matchable 1.157E+02, num_unmatchable 3.814E+02, confidence 2.078E-01, row_norm 9.258E-01}
[06/14/2024 00:34:38 gluefactory INFO] [E 6 | it 900] loss {total 8.117E-01, last 3.466E-01, assignment_nll 3.466E-01, nll_pos 4.773E-01, nll_neg 2.159E-01, num_matchable 1.227E+02, num_unmatchable 3.767E+02, confidence 2.055E-01, row_norm 9.482E-01}
[06/14/2024 00:38:01 gluefactory INFO] [E 6 | it 1000] loss {total 6.994E-01, last 2.005E-01, assignment_nll 2.005E-01, nll_pos 1.940E-01, nll_neg 2.070E-01, num_matchable 1.383E+02, num_unmatchable 3.579E+02, confidence 2.087E-01, row_norm 9.587E-01}
[06/14/2024 00:43:58 gluefactory INFO] [Validation] {match_recall 9.289E-01, match_precision 7.182E-01, accuracy 9.235E-01, average_precision 6.769E-01, loss/total 2.912E-01, loss/last 2.912E-01, loss/assignment_nll 2.912E-01, loss/nll_pos 3.750E-01, loss/nll_neg 2.074E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.436E-01}
[06/14/2024 00:43:58 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 00:43:58 gluefactory INFO] New best val: loss/total=0.29116573108493204
[06/14/2024 00:47:25 gluefactory INFO] [E 6 | it 1100] loss {total 8.768E-01, last 3.823E-01, assignment_nll 3.823E-01, nll_pos 5.328E-01, nll_neg 2.317E-01, num_matchable 1.358E+02, num_unmatchable 3.653E+02, confidence 2.039E-01, row_norm 9.609E-01}
[06/14/2024 00:50:50 gluefactory INFO] [E 6 | it 1200] loss {total 9.639E-01, last 4.554E-01, assignment_nll 4.554E-01, nll_pos 6.738E-01, nll_neg 2.369E-01, num_matchable 1.232E+02, num_unmatchable 3.724E+02, confidence 2.052E-01, row_norm 9.309E-01}
[06/14/2024 00:54:16 gluefactory INFO] [E 6 | it 1300] loss {total 9.662E-01, last 3.878E-01, assignment_nll 3.878E-01, nll_pos 5.501E-01, nll_neg 2.255E-01, num_matchable 1.201E+02, num_unmatchable 3.761E+02, confidence 2.186E-01, row_norm 9.348E-01}
[06/14/2024 00:57:41 gluefactory INFO] [E 6 | it 1400] loss {total 7.903E-01, last 2.375E-01, assignment_nll 2.375E-01, nll_pos 2.616E-01, nll_neg 2.133E-01, num_matchable 1.398E+02, num_unmatchable 3.578E+02, confidence 2.336E-01, row_norm 9.443E-01}
[06/14/2024 01:01:06 gluefactory INFO] [E 6 | it 1500] loss {total 7.363E-01, last 2.484E-01, assignment_nll 2.484E-01, nll_pos 2.984E-01, nll_neg 1.985E-01, num_matchable 1.452E+02, num_unmatchable 3.505E+02, confidence 2.112E-01, row_norm 9.498E-01}
[06/14/2024 01:07:05 gluefactory INFO] [Validation] {match_recall 9.244E-01, match_precision 7.236E-01, accuracy 9.263E-01, average_precision 6.801E-01, loss/total 3.059E-01, loss/last 3.059E-01, loss/assignment_nll 3.059E-01, loss/nll_pos 4.034E-01, loss/nll_neg 2.084E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.453E-01}
[06/14/2024 01:14:36 gluefactory INFO] [Validation] {match_recall 9.224E-01, match_precision 7.124E-01, accuracy 9.213E-01, average_precision 6.694E-01, loss/total 3.234E-01, loss/last 3.234E-01, loss/assignment_nll 3.234E-01, loss/nll_pos 4.133E-01, loss/nll_neg 2.335E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.416E-01}
[06/14/2024 01:14:38 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_6_10814.tar
[06/14/2024 01:14:39 gluefactory INFO] Starting epoch 7
[06/14/2024 01:14:39 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 01:14:45 gluefactory INFO] [E 7 | it 0] loss {total 9.645E-01, last 4.058E-01, assignment_nll 4.058E-01, nll_pos 5.537E-01, nll_neg 2.579E-01, num_matchable 1.174E+02, num_unmatchable 3.828E+02, confidence 2.127E-01, row_norm 9.349E-01}
[06/14/2024 01:18:10 gluefactory INFO] [E 7 | it 100] loss {total 7.337E-01, last 2.182E-01, assignment_nll 2.182E-01, nll_pos 2.483E-01, nll_neg 1.880E-01, num_matchable 1.283E+02, num_unmatchable 3.702E+02, confidence 2.033E-01, row_norm 9.526E-01}
[06/14/2024 01:21:35 gluefactory INFO] [E 7 | it 200] loss {total 7.147E-01, last 1.866E-01, assignment_nll 1.866E-01, nll_pos 1.977E-01, nll_neg 1.756E-01, num_matchable 1.328E+02, num_unmatchable 3.622E+02, confidence 2.065E-01, row_norm 9.611E-01}
[06/14/2024 01:25:00 gluefactory INFO] [E 7 | it 300] loss {total 1.033E+00, last 4.106E-01, assignment_nll 4.106E-01, nll_pos 6.352E-01, nll_neg 1.860E-01, num_matchable 1.149E+02, num_unmatchable 3.824E+02, confidence 2.132E-01, row_norm 9.445E-01}
[06/14/2024 01:28:25 gluefactory INFO] [E 7 | it 400] loss {total 8.032E-01, last 3.036E-01, assignment_nll 3.036E-01, nll_pos 4.086E-01, nll_neg 1.986E-01, num_matchable 1.109E+02, num_unmatchable 3.874E+02, confidence 2.043E-01, row_norm 9.466E-01}
[06/14/2024 01:31:50 gluefactory INFO] [E 7 | it 500] loss {total 8.053E-01, last 2.898E-01, assignment_nll 2.898E-01, nll_pos 3.559E-01, nll_neg 2.237E-01, num_matchable 1.272E+02, num_unmatchable 3.730E+02, confidence 2.011E-01, row_norm 9.496E-01}
[06/14/2024 01:37:48 gluefactory INFO] [Validation] {match_recall 9.293E-01, match_precision 7.211E-01, accuracy 9.256E-01, average_precision 6.808E-01, loss/total 2.904E-01, loss/last 2.904E-01, loss/assignment_nll 2.904E-01, loss/nll_pos 3.654E-01, loss/nll_neg 2.154E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.432E-01}
[06/14/2024 01:37:48 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 01:37:49 gluefactory INFO] New best val: loss/total=0.2904030154903257
[06/14/2024 01:41:15 gluefactory INFO] [E 7 | it 600] loss {total 8.509E-01, last 2.729E-01, assignment_nll 2.729E-01, nll_pos 3.226E-01, nll_neg 2.233E-01, num_matchable 1.247E+02, num_unmatchable 3.696E+02, confidence 2.273E-01, row_norm 9.491E-01}
[06/14/2024 01:44:40 gluefactory INFO] [E 7 | it 700] loss {total 7.978E-01, last 2.888E-01, assignment_nll 2.888E-01, nll_pos 3.723E-01, nll_neg 2.052E-01, num_matchable 1.190E+02, num_unmatchable 3.792E+02, confidence 2.023E-01, row_norm 9.401E-01}
[06/14/2024 01:48:05 gluefactory INFO] [E 7 | it 800] loss {total 9.551E-01, last 3.866E-01, assignment_nll 3.866E-01, nll_pos 5.509E-01, nll_neg 2.223E-01, num_matchable 1.186E+02, num_unmatchable 3.783E+02, confidence 2.134E-01, row_norm 9.362E-01}
[06/14/2024 01:51:30 gluefactory INFO] [E 7 | it 900] loss {total 7.652E-01, last 2.240E-01, assignment_nll 2.240E-01, nll_pos 2.619E-01, nll_neg 1.862E-01, num_matchable 1.238E+02, num_unmatchable 3.740E+02, confidence 2.183E-01, row_norm 9.533E-01}
[06/14/2024 01:54:55 gluefactory INFO] [E 7 | it 1000] loss {total 6.572E-01, last 1.823E-01, assignment_nll 1.823E-01, nll_pos 2.028E-01, nll_neg 1.619E-01, num_matchable 1.349E+02, num_unmatchable 3.625E+02, confidence 1.915E-01, row_norm 9.616E-01}
[06/14/2024 02:00:54 gluefactory INFO] [Validation] {match_recall 9.321E-01, match_precision 7.300E-01, accuracy 9.289E-01, average_precision 6.904E-01, loss/total 2.751E-01, loss/last 2.751E-01, loss/assignment_nll 2.751E-01, loss/nll_pos 3.514E-01, loss/nll_neg 1.987E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.459E-01}
[06/14/2024 02:00:54 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 02:00:55 gluefactory INFO] New best val: loss/total=0.27506410586721136
[06/14/2024 02:04:21 gluefactory INFO] [E 7 | it 1100] loss {total 6.854E-01, last 2.018E-01, assignment_nll 2.018E-01, nll_pos 2.515E-01, nll_neg 1.520E-01, num_matchable 1.276E+02, num_unmatchable 3.723E+02, confidence 2.026E-01, row_norm 9.584E-01}
[06/14/2024 02:07:46 gluefactory INFO] [E 7 | it 1200] loss {total 7.321E-01, last 2.018E-01, assignment_nll 2.018E-01, nll_pos 2.191E-01, nll_neg 1.845E-01, num_matchable 1.386E+02, num_unmatchable 3.556E+02, confidence 2.140E-01, row_norm 9.620E-01}
[06/14/2024 02:11:12 gluefactory INFO] [E 7 | it 1300] loss {total 8.601E-01, last 2.658E-01, assignment_nll 2.658E-01, nll_pos 3.093E-01, nll_neg 2.223E-01, num_matchable 1.302E+02, num_unmatchable 3.666E+02, confidence 2.322E-01, row_norm 9.501E-01}
[06/14/2024 02:14:37 gluefactory INFO] [E 7 | it 1400] loss {total 6.851E-01, last 1.943E-01, assignment_nll 1.943E-01, nll_pos 2.301E-01, nll_neg 1.585E-01, num_matchable 1.438E+02, num_unmatchable 3.548E+02, confidence 2.055E-01, row_norm 9.610E-01}
[06/14/2024 02:18:02 gluefactory INFO] [E 7 | it 1500] loss {total 7.709E-01, last 2.571E-01, assignment_nll 2.571E-01, nll_pos 3.391E-01, nll_neg 1.750E-01, num_matchable 1.361E+02, num_unmatchable 3.598E+02, confidence 2.069E-01, row_norm 9.560E-01}
[06/14/2024 02:24:01 gluefactory INFO] [Validation] {match_recall 9.334E-01, match_precision 7.412E-01, accuracy 9.339E-01, average_precision 7.012E-01, loss/total 2.678E-01, loss/last 2.678E-01, loss/assignment_nll 2.678E-01, loss/nll_pos 3.576E-01, loss/nll_neg 1.779E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.521E-01}
[06/14/2024 02:24:01 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 02:24:01 gluefactory INFO] New best val: loss/total=0.26776263849409154
[06/14/2024 02:31:31 gluefactory INFO] [Validation] {match_recall 9.318E-01, match_precision 7.322E-01, accuracy 9.299E-01, average_precision 6.925E-01, loss/total 2.780E-01, loss/last 2.780E-01, loss/assignment_nll 2.780E-01, loss/nll_pos 3.492E-01, loss/nll_neg 2.067E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.457E-01}
[06/14/2024 02:31:33 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_7_12359.tar
[06/14/2024 02:31:34 gluefactory INFO] Starting epoch 8
[06/14/2024 02:31:34 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 02:31:39 gluefactory INFO] [E 8 | it 0] loss {total 7.351E-01, last 2.056E-01, assignment_nll 2.056E-01, nll_pos 2.161E-01, nll_neg 1.952E-01, num_matchable 1.341E+02, num_unmatchable 3.661E+02, confidence 2.102E-01, row_norm 9.567E-01}
[06/14/2024 02:35:04 gluefactory INFO] [E 8 | it 100] loss {total 9.854E-01, last 3.345E-01, assignment_nll 3.345E-01, nll_pos 4.115E-01, nll_neg 2.576E-01, num_matchable 1.268E+02, num_unmatchable 3.706E+02, confidence 2.507E-01, row_norm 9.270E-01}
[06/14/2024 02:38:28 gluefactory INFO] [E 8 | it 200] loss {total 6.571E-01, last 1.712E-01, assignment_nll 1.712E-01, nll_pos 1.756E-01, nll_neg 1.668E-01, num_matchable 1.286E+02, num_unmatchable 3.662E+02, confidence 2.015E-01, row_norm 9.631E-01}
[06/14/2024 02:41:51 gluefactory INFO] [E 8 | it 300] loss {total 7.615E-01, last 1.914E-01, assignment_nll 1.914E-01, nll_pos 2.166E-01, nll_neg 1.661E-01, num_matchable 1.251E+02, num_unmatchable 3.720E+02, confidence 2.140E-01, row_norm 9.562E-01}
[06/14/2024 02:45:15 gluefactory INFO] [E 8 | it 400] loss {total 9.166E-01, last 3.331E-01, assignment_nll 3.331E-01, nll_pos 4.442E-01, nll_neg 2.219E-01, num_matchable 1.236E+02, num_unmatchable 3.740E+02, confidence 2.132E-01, row_norm 9.312E-01}
[06/14/2024 02:48:38 gluefactory INFO] [E 8 | it 500] loss {total 6.620E-01, last 1.824E-01, assignment_nll 1.824E-01, nll_pos 1.934E-01, nll_neg 1.714E-01, num_matchable 1.305E+02, num_unmatchable 3.692E+02, confidence 1.993E-01, row_norm 9.648E-01}
[06/14/2024 02:54:36 gluefactory INFO] [Validation] {match_recall 9.373E-01, match_precision 7.394E-01, accuracy 9.332E-01, average_precision 7.020E-01, loss/total 2.563E-01, loss/last 2.563E-01, loss/assignment_nll 2.563E-01, loss/nll_pos 3.278E-01, loss/nll_neg 1.848E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.504E-01}
[06/14/2024 02:54:36 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 02:54:36 gluefactory INFO] New best val: loss/total=0.2562636312404362
[06/14/2024 02:58:03 gluefactory INFO] [E 8 | it 600] loss {total 7.373E-01, last 2.138E-01, assignment_nll 2.138E-01, nll_pos 2.385E-01, nll_neg 1.890E-01, num_matchable 1.164E+02, num_unmatchable 3.787E+02, confidence 2.098E-01, row_norm 9.609E-01}
[06/14/2024 03:01:29 gluefactory INFO] [E 8 | it 700] loss {total 6.992E-01, last 2.099E-01, assignment_nll 2.099E-01, nll_pos 2.392E-01, nll_neg 1.805E-01, num_matchable 1.298E+02, num_unmatchable 3.702E+02, confidence 1.907E-01, row_norm 9.582E-01}
[06/14/2024 03:04:54 gluefactory INFO] [E 8 | it 800] loss {total 7.761E-01, last 2.306E-01, assignment_nll 2.306E-01, nll_pos 2.596E-01, nll_neg 2.017E-01, num_matchable 1.175E+02, num_unmatchable 3.790E+02, confidence 2.093E-01, row_norm 9.557E-01}
[06/14/2024 03:08:20 gluefactory INFO] [E 8 | it 900] loss {total 7.619E-01, last 2.719E-01, assignment_nll 2.719E-01, nll_pos 3.611E-01, nll_neg 1.826E-01, num_matchable 1.167E+02, num_unmatchable 3.821E+02, confidence 2.003E-01, row_norm 9.559E-01}
[06/14/2024 03:11:45 gluefactory INFO] [E 8 | it 1000] loss {total 6.012E-01, last 1.421E-01, assignment_nll 1.421E-01, nll_pos 1.199E-01, nll_neg 1.644E-01, num_matchable 1.393E+02, num_unmatchable 3.591E+02, confidence 1.887E-01, row_norm 9.667E-01}
[06/14/2024 03:17:44 gluefactory INFO] [Validation] {match_recall 9.386E-01, match_precision 7.348E-01, accuracy 9.312E-01, average_precision 6.986E-01, loss/total 2.579E-01, loss/last 2.579E-01, loss/assignment_nll 2.579E-01, loss/nll_pos 3.154E-01, loss/nll_neg 2.004E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.517E-01}
[06/14/2024 03:21:11 gluefactory INFO] [E 8 | it 1100] loss {total 7.988E-01, last 2.395E-01, assignment_nll 2.395E-01, nll_pos 2.889E-01, nll_neg 1.900E-01, num_matchable 1.276E+02, num_unmatchable 3.715E+02, confidence 2.219E-01, row_norm 9.585E-01}
[06/14/2024 03:24:36 gluefactory INFO] [E 8 | it 1200] loss {total 6.883E-01, last 1.919E-01, assignment_nll 1.919E-01, nll_pos 2.316E-01, nll_neg 1.521E-01, num_matchable 1.244E+02, num_unmatchable 3.721E+02, confidence 2.006E-01, row_norm 9.641E-01}
[06/14/2024 03:28:02 gluefactory INFO] [E 8 | it 1300] loss {total 7.377E-01, last 2.192E-01, assignment_nll 2.192E-01, nll_pos 2.765E-01, nll_neg 1.620E-01, num_matchable 1.304E+02, num_unmatchable 3.675E+02, confidence 2.063E-01, row_norm 9.542E-01}
[06/14/2024 03:31:27 gluefactory INFO] [E 8 | it 1400] loss {total 7.219E-01, last 2.211E-01, assignment_nll 2.211E-01, nll_pos 2.925E-01, nll_neg 1.496E-01, num_matchable 1.468E+02, num_unmatchable 3.524E+02, confidence 2.023E-01, row_norm 9.538E-01}
[06/14/2024 03:34:53 gluefactory INFO] [E 8 | it 1500] loss {total 6.674E-01, last 1.631E-01, assignment_nll 1.631E-01, nll_pos 1.688E-01, nll_neg 1.574E-01, num_matchable 1.457E+02, num_unmatchable 3.517E+02, confidence 2.108E-01, row_norm 9.657E-01}
[06/14/2024 03:40:52 gluefactory INFO] [Validation] {match_recall 9.329E-01, match_precision 7.394E-01, accuracy 9.330E-01, average_precision 7.002E-01, loss/total 2.749E-01, loss/last 2.749E-01, loss/assignment_nll 2.749E-01, loss/nll_pos 3.540E-01, loss/nll_neg 1.958E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.529E-01}
[06/14/2024 03:48:25 gluefactory INFO] [Validation] {match_recall 9.357E-01, match_precision 7.507E-01, accuracy 9.376E-01, average_precision 7.119E-01, loss/total 2.566E-01, loss/last 2.566E-01, loss/assignment_nll 2.566E-01, loss/nll_pos 3.339E-01, loss/nll_neg 1.793E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.538E-01}
[06/14/2024 03:48:26 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_8_13904.tar
[06/14/2024 03:48:27 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_0_1544.tar
[06/14/2024 03:48:27 gluefactory INFO] Starting epoch 9
[06/14/2024 03:48:27 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 03:48:32 gluefactory INFO] [E 9 | it 0] loss {total 7.585E-01, last 2.232E-01, assignment_nll 2.232E-01, nll_pos 2.573E-01, nll_neg 1.892E-01, num_matchable 1.242E+02, num_unmatchable 3.743E+02, confidence 2.092E-01, row_norm 9.575E-01}
[06/14/2024 03:51:58 gluefactory INFO] [E 9 | it 100] loss {total 7.601E-01, last 2.395E-01, assignment_nll 2.395E-01, nll_pos 3.189E-01, nll_neg 1.601E-01, num_matchable 1.250E+02, num_unmatchable 3.712E+02, confidence 2.078E-01, row_norm 9.551E-01}
[06/14/2024 03:55:23 gluefactory INFO] [E 9 | it 200] loss {total 7.294E-01, last 2.462E-01, assignment_nll 2.462E-01, nll_pos 3.305E-01, nll_neg 1.619E-01, num_matchable 1.306E+02, num_unmatchable 3.662E+02, confidence 1.966E-01, row_norm 9.635E-01}
[06/14/2024 03:58:48 gluefactory INFO] [E 9 | it 300] loss {total 8.502E-01, last 2.860E-01, assignment_nll 2.860E-01, nll_pos 3.475E-01, nll_neg 2.244E-01, num_matchable 1.194E+02, num_unmatchable 3.794E+02, confidence 2.103E-01, row_norm 9.308E-01}
[06/14/2024 04:02:13 gluefactory INFO] [E 9 | it 400] loss {total 9.241E-01, last 3.073E-01, assignment_nll 3.073E-01, nll_pos 3.940E-01, nll_neg 2.206E-01, num_matchable 1.078E+02, num_unmatchable 3.907E+02, confidence 2.194E-01, row_norm 9.249E-01}
[06/14/2024 04:05:39 gluefactory INFO] [E 9 | it 500] loss {total 6.539E-01, last 1.549E-01, assignment_nll 1.549E-01, nll_pos 1.631E-01, nll_neg 1.467E-01, num_matchable 1.338E+02, num_unmatchable 3.640E+02, confidence 2.092E-01, row_norm 9.658E-01}
[06/14/2024 04:11:38 gluefactory INFO] [Validation] {match_recall 9.370E-01, match_precision 7.548E-01, accuracy 9.397E-01, average_precision 7.162E-01, loss/total 2.514E-01, loss/last 2.514E-01, loss/assignment_nll 2.514E-01, loss/nll_pos 3.458E-01, loss/nll_neg 1.569E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.533E-01}
[06/14/2024 04:11:38 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 04:11:39 gluefactory INFO] New best val: loss/total=0.25138133075315766
[06/14/2024 04:15:06 gluefactory INFO] [E 9 | it 600] loss {total 6.460E-01, last 1.664E-01, assignment_nll 1.664E-01, nll_pos 1.724E-01, nll_neg 1.603E-01, num_matchable 1.160E+02, num_unmatchable 3.803E+02, confidence 1.965E-01, row_norm 9.647E-01}
[06/14/2024 04:18:31 gluefactory INFO] [E 9 | it 700] loss {total 8.202E-01, last 2.359E-01, assignment_nll 2.359E-01, nll_pos 3.028E-01, nll_neg 1.690E-01, num_matchable 1.280E+02, num_unmatchable 3.694E+02, confidence 2.073E-01, row_norm 9.549E-01}
[06/14/2024 04:21:56 gluefactory INFO] [E 9 | it 800] loss {total 8.517E-01, last 2.738E-01, assignment_nll 2.738E-01, nll_pos 3.678E-01, nll_neg 1.799E-01, num_matchable 1.111E+02, num_unmatchable 3.833E+02, confidence 2.093E-01, row_norm 9.442E-01}
[06/14/2024 04:25:22 gluefactory INFO] [E 9 | it 900] loss {total 7.463E-01, last 1.911E-01, assignment_nll 1.911E-01, nll_pos 1.923E-01, nll_neg 1.899E-01, num_matchable 1.376E+02, num_unmatchable 3.610E+02, confidence 2.212E-01, row_norm 9.604E-01}
[06/14/2024 04:28:47 gluefactory INFO] [E 9 | it 1000] loss {total 6.840E-01, last 1.857E-01, assignment_nll 1.857E-01, nll_pos 2.172E-01, nll_neg 1.542E-01, num_matchable 1.420E+02, num_unmatchable 3.540E+02, confidence 2.006E-01, row_norm 9.550E-01}
[06/14/2024 04:34:47 gluefactory INFO] [Validation] {match_recall 9.374E-01, match_precision 7.500E-01, accuracy 9.374E-01, average_precision 7.127E-01, loss/total 2.544E-01, loss/last 2.544E-01, loss/assignment_nll 2.544E-01, loss/nll_pos 3.359E-01, loss/nll_neg 1.729E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.518E-01}
[06/14/2024 04:38:03 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_9_15000.tar
[06/14/2024 04:38:04 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_1_3089.tar
[06/14/2024 04:38:15 gluefactory INFO] [E 9 | it 1100] loss {total 7.177E-01, last 2.286E-01, assignment_nll 2.286E-01, nll_pos 3.175E-01, nll_neg 1.398E-01, num_matchable 1.226E+02, num_unmatchable 3.771E+02, confidence 1.964E-01, row_norm 9.631E-01}
[06/14/2024 04:41:40 gluefactory INFO] [E 9 | it 1200] loss {total 7.511E-01, last 2.441E-01, assignment_nll 2.441E-01, nll_pos 3.350E-01, nll_neg 1.533E-01, num_matchable 1.184E+02, num_unmatchable 3.774E+02, confidence 1.999E-01, row_norm 9.537E-01}
[06/14/2024 04:45:05 gluefactory INFO] [E 9 | it 1300] loss {total 7.218E-01, last 2.045E-01, assignment_nll 2.045E-01, nll_pos 2.415E-01, nll_neg 1.675E-01, num_matchable 1.337E+02, num_unmatchable 3.646E+02, confidence 2.100E-01, row_norm 9.623E-01}
[06/14/2024 04:48:28 gluefactory INFO] [E 9 | it 1400] loss {total 7.252E-01, last 2.198E-01, assignment_nll 2.198E-01, nll_pos 2.978E-01, nll_neg 1.417E-01, num_matchable 1.340E+02, num_unmatchable 3.654E+02, confidence 1.966E-01, row_norm 9.571E-01}
[06/14/2024 04:51:52 gluefactory INFO] [E 9 | it 1500] loss {total 6.648E-01, last 1.702E-01, assignment_nll 1.702E-01, nll_pos 1.944E-01, nll_neg 1.460E-01, num_matchable 1.445E+02, num_unmatchable 3.519E+02, confidence 2.048E-01, row_norm 9.636E-01}
[06/14/2024 04:57:48 gluefactory INFO] [Validation] {match_recall 9.383E-01, match_precision 7.575E-01, accuracy 9.404E-01, average_precision 7.202E-01, loss/total 2.456E-01, loss/last 2.456E-01, loss/assignment_nll 2.456E-01, loss/nll_pos 3.276E-01, loss/nll_neg 1.637E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.523E-01}
[06/14/2024 04:57:48 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 04:57:49 gluefactory INFO] New best val: loss/total=0.2456436029560867
[06/14/2024 05:05:19 gluefactory INFO] [Validation] {match_recall 9.405E-01, match_precision 7.635E-01, accuracy 9.431E-01, average_precision 7.265E-01, loss/total 2.343E-01, loss/last 2.343E-01, loss/assignment_nll 2.343E-01, loss/nll_pos 3.158E-01, loss/nll_neg 1.528E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.557E-01}
[06/14/2024 05:05:19 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 05:05:20 gluefactory INFO] New best val: loss/total=0.23430101043977639
[06/14/2024 05:05:22 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_9_15449.tar
[06/14/2024 05:05:22 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_2_4634.tar
[06/14/2024 05:05:22 gluefactory INFO] Starting epoch 10
[06/14/2024 05:05:22 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 05:05:28 gluefactory INFO] [E 10 | it 0] loss {total 7.948E-01, last 2.747E-01, assignment_nll 2.747E-01, nll_pos 3.889E-01, nll_neg 1.605E-01, num_matchable 1.244E+02, num_unmatchable 3.737E+02, confidence 1.986E-01, row_norm 9.530E-01}
[06/14/2024 05:08:53 gluefactory INFO] [E 10 | it 100] loss {total 6.784E-01, last 1.652E-01, assignment_nll 1.652E-01, nll_pos 1.858E-01, nll_neg 1.446E-01, num_matchable 1.311E+02, num_unmatchable 3.658E+02, confidence 2.090E-01, row_norm 9.606E-01}
[06/14/2024 05:12:19 gluefactory INFO] [E 10 | it 200] loss {total 6.808E-01, last 1.667E-01, assignment_nll 1.667E-01, nll_pos 1.919E-01, nll_neg 1.415E-01, num_matchable 1.300E+02, num_unmatchable 3.649E+02, confidence 1.966E-01, row_norm 9.595E-01}
[06/14/2024 05:15:44 gluefactory INFO] [E 10 | it 300] loss {total 7.595E-01, last 2.075E-01, assignment_nll 2.075E-01, nll_pos 2.587E-01, nll_neg 1.564E-01, num_matchable 1.208E+02, num_unmatchable 3.776E+02, confidence 2.032E-01, row_norm 9.539E-01}
[06/14/2024 05:19:10 gluefactory INFO] [E 10 | it 400] loss {total 7.335E-01, last 1.995E-01, assignment_nll 1.995E-01, nll_pos 2.485E-01, nll_neg 1.504E-01, num_matchable 1.229E+02, num_unmatchable 3.749E+02, confidence 2.007E-01, row_norm 9.600E-01}
[06/14/2024 05:22:35 gluefactory INFO] [E 10 | it 500] loss {total 6.624E-01, last 1.740E-01, assignment_nll 1.740E-01, nll_pos 1.910E-01, nll_neg 1.570E-01, num_matchable 1.182E+02, num_unmatchable 3.826E+02, confidence 1.862E-01, row_norm 9.665E-01}
[06/14/2024 05:28:34 gluefactory INFO] [Validation] {match_recall 9.420E-01, match_precision 7.599E-01, accuracy 9.416E-01, average_precision 7.240E-01, loss/total 2.311E-01, loss/last 2.311E-01, loss/assignment_nll 2.311E-01, loss/nll_pos 3.105E-01, loss/nll_neg 1.516E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.523E-01}
[06/14/2024 05:28:34 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 05:28:35 gluefactory INFO] New best val: loss/total=0.23105418856626264
[06/14/2024 05:32:02 gluefactory INFO] [E 10 | it 600] loss {total 6.789E-01, last 1.709E-01, assignment_nll 1.709E-01, nll_pos 1.886E-01, nll_neg 1.532E-01, num_matchable 1.227E+02, num_unmatchable 3.743E+02, confidence 2.075E-01, row_norm 9.667E-01}
[06/14/2024 05:35:27 gluefactory INFO] [E 10 | it 700] loss {total 6.635E-01, last 1.680E-01, assignment_nll 1.680E-01, nll_pos 1.932E-01, nll_neg 1.429E-01, num_matchable 1.246E+02, num_unmatchable 3.746E+02, confidence 1.872E-01, row_norm 9.639E-01}
[06/14/2024 05:38:53 gluefactory INFO] [E 10 | it 800] loss {total 8.068E-01, last 2.472E-01, assignment_nll 2.472E-01, nll_pos 3.002E-01, nll_neg 1.941E-01, num_matchable 1.122E+02, num_unmatchable 3.839E+02, confidence 2.107E-01, row_norm 9.455E-01}
[06/14/2024 05:42:18 gluefactory INFO] [E 10 | it 900] loss {total 8.174E-01, last 2.839E-01, assignment_nll 2.839E-01, nll_pos 3.964E-01, nll_neg 1.715E-01, num_matchable 1.386E+02, num_unmatchable 3.585E+02, confidence 2.096E-01, row_norm 9.410E-01}
[06/14/2024 05:45:43 gluefactory INFO] [E 10 | it 1000] loss {total 7.242E-01, last 2.229E-01, assignment_nll 2.229E-01, nll_pos 2.705E-01, nll_neg 1.754E-01, num_matchable 1.336E+02, num_unmatchable 3.626E+02, confidence 2.036E-01, row_norm 9.559E-01}
[06/14/2024 05:51:43 gluefactory INFO] [Validation] {match_recall 9.390E-01, match_precision 7.390E-01, accuracy 9.331E-01, average_precision 7.044E-01, loss/total 2.555E-01, loss/last 2.555E-01, loss/assignment_nll 2.555E-01, loss/nll_pos 3.038E-01, loss/nll_neg 2.072E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.450E-01}
[06/14/2024 05:55:10 gluefactory INFO] [E 10 | it 1100] loss {total 6.833E-01, last 2.056E-01, assignment_nll 2.056E-01, nll_pos 2.575E-01, nll_neg 1.536E-01, num_matchable 1.205E+02, num_unmatchable 3.789E+02, confidence 1.960E-01, row_norm 9.665E-01}
[06/14/2024 05:58:35 gluefactory INFO] [E 10 | it 1200] loss {total 7.643E-01, last 2.640E-01, assignment_nll 2.640E-01, nll_pos 3.769E-01, nll_neg 1.512E-01, num_matchable 1.258E+02, num_unmatchable 3.704E+02, confidence 1.949E-01, row_norm 9.522E-01}
[06/14/2024 06:02:01 gluefactory INFO] [E 10 | it 1300] loss {total 7.253E-01, last 1.951E-01, assignment_nll 1.951E-01, nll_pos 2.423E-01, nll_neg 1.480E-01, num_matchable 1.341E+02, num_unmatchable 3.627E+02, confidence 2.094E-01, row_norm 9.600E-01}
[06/14/2024 06:05:26 gluefactory INFO] [E 10 | it 1400] loss {total 6.717E-01, last 1.768E-01, assignment_nll 1.768E-01, nll_pos 2.065E-01, nll_neg 1.471E-01, num_matchable 1.358E+02, num_unmatchable 3.628E+02, confidence 1.934E-01, row_norm 9.692E-01}
[06/14/2024 06:08:52 gluefactory INFO] [E 10 | it 1500] loss {total 6.244E-01, last 1.489E-01, assignment_nll 1.489E-01, nll_pos 1.649E-01, nll_neg 1.328E-01, num_matchable 1.423E+02, num_unmatchable 3.539E+02, confidence 2.020E-01, row_norm 9.640E-01}
[06/14/2024 06:14:50 gluefactory INFO] [Validation] {match_recall 9.440E-01, match_precision 7.599E-01, accuracy 9.418E-01, average_precision 7.253E-01, loss/total 2.243E-01, loss/last 2.243E-01, loss/assignment_nll 2.243E-01, loss/nll_pos 2.930E-01, loss/nll_neg 1.557E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.551E-01}
[06/14/2024 06:14:50 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 06:14:51 gluefactory INFO] New best val: loss/total=0.22433748926380098
[06/14/2024 06:22:23 gluefactory INFO] [Validation] {match_recall 9.459E-01, match_precision 7.732E-01, accuracy 9.466E-01, average_precision 7.387E-01, loss/total 2.139E-01, loss/last 2.139E-01, loss/assignment_nll 2.139E-01, loss/nll_pos 2.824E-01, loss/nll_neg 1.454E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.586E-01}
[06/14/2024 06:22:23 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 06:22:24 gluefactory INFO] New best val: loss/total=0.21387345648348008
[06/14/2024 06:22:25 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_10_16994.tar
[06/14/2024 06:22:26 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_3_5000.tar
[06/14/2024 06:22:26 gluefactory INFO] Starting epoch 11
[06/14/2024 06:22:26 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 06:22:31 gluefactory INFO] [E 11 | it 0] loss {total 6.796E-01, last 1.807E-01, assignment_nll 1.807E-01, nll_pos 2.096E-01, nll_neg 1.517E-01, num_matchable 1.285E+02, num_unmatchable 3.685E+02, confidence 2.039E-01, row_norm 9.647E-01}
[06/14/2024 06:25:57 gluefactory INFO] [E 11 | it 100] loss {total 7.062E-01, last 1.862E-01, assignment_nll 1.862E-01, nll_pos 2.043E-01, nll_neg 1.682E-01, num_matchable 1.275E+02, num_unmatchable 3.703E+02, confidence 2.117E-01, row_norm 9.604E-01}
[06/14/2024 06:29:22 gluefactory INFO] [E 11 | it 200] loss {total 7.884E-01, last 2.709E-01, assignment_nll 2.709E-01, nll_pos 3.975E-01, nll_neg 1.444E-01, num_matchable 1.353E+02, num_unmatchable 3.596E+02, confidence 1.988E-01, row_norm 9.571E-01}
[06/14/2024 06:32:48 gluefactory INFO] [E 11 | it 300] loss {total 1.009E+00, last 4.503E-01, assignment_nll 4.503E-01, nll_pos 7.122E-01, nll_neg 1.883E-01, num_matchable 1.234E+02, num_unmatchable 3.740E+02, confidence 2.085E-01, row_norm 9.308E-01}
[06/14/2024 06:36:13 gluefactory INFO] [E 11 | it 400] loss {total 7.127E-01, last 2.129E-01, assignment_nll 2.129E-01, nll_pos 2.699E-01, nll_neg 1.559E-01, num_matchable 1.120E+02, num_unmatchable 3.866E+02, confidence 1.884E-01, row_norm 9.566E-01}
[06/14/2024 06:39:38 gluefactory INFO] [E 11 | it 500] loss {total 6.274E-01, last 1.382E-01, assignment_nll 1.382E-01, nll_pos 1.287E-01, nll_neg 1.478E-01, num_matchable 1.240E+02, num_unmatchable 3.751E+02, confidence 1.928E-01, row_norm 9.701E-01}
[06/14/2024 06:45:37 gluefactory INFO] [Validation] {match_recall 9.455E-01, match_precision 7.644E-01, accuracy 9.436E-01, average_precision 7.300E-01, loss/total 2.201E-01, loss/last 2.201E-01, loss/assignment_nll 2.201E-01, loss/nll_pos 2.900E-01, loss/nll_neg 1.502E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.604E-01}
[06/14/2024 06:48:59 gluefactory INFO] [E 11 | it 600] loss {total 6.818E-01, last 1.832E-01, assignment_nll 1.832E-01, nll_pos 2.144E-01, nll_neg 1.520E-01, num_matchable 1.276E+02, num_unmatchable 3.672E+02, confidence 2.088E-01, row_norm 9.675E-01}
[06/14/2024 06:52:21 gluefactory INFO] [E 11 | it 700] loss {total 6.190E-01, last 1.337E-01, assignment_nll 1.337E-01, nll_pos 1.375E-01, nll_neg 1.299E-01, num_matchable 1.280E+02, num_unmatchable 3.710E+02, confidence 1.941E-01, row_norm 9.707E-01}
[06/14/2024 06:55:42 gluefactory INFO] [E 11 | it 800] loss {total 7.259E-01, last 2.008E-01, assignment_nll 2.008E-01, nll_pos 2.511E-01, nll_neg 1.505E-01, num_matchable 1.259E+02, num_unmatchable 3.695E+02, confidence 2.081E-01, row_norm 9.597E-01}
[06/14/2024 06:59:03 gluefactory INFO] [E 11 | it 900] loss {total 6.881E-01, last 1.816E-01, assignment_nll 1.816E-01, nll_pos 2.131E-01, nll_neg 1.502E-01, num_matchable 1.189E+02, num_unmatchable 3.792E+02, confidence 1.983E-01, row_norm 9.541E-01}
[06/14/2024 07:02:26 gluefactory INFO] [E 11 | it 1000] loss {total 5.623E-01, last 1.192E-01, assignment_nll 1.192E-01, nll_pos 1.177E-01, nll_neg 1.207E-01, num_matchable 1.421E+02, num_unmatchable 3.540E+02, confidence 1.931E-01, row_norm 9.705E-01}
[06/14/2024 07:08:21 gluefactory INFO] [Validation] {match_recall 9.428E-01, match_precision 7.726E-01, accuracy 9.466E-01, average_precision 7.375E-01, loss/total 2.267E-01, loss/last 2.267E-01, loss/assignment_nll 2.267E-01, loss/nll_pos 3.054E-01, loss/nll_neg 1.479E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.545E-01}
[06/14/2024 07:11:46 gluefactory INFO] [E 11 | it 1100] loss {total 8.141E-01, last 2.808E-01, assignment_nll 2.808E-01, nll_pos 3.759E-01, nll_neg 1.858E-01, num_matchable 1.368E+02, num_unmatchable 3.583E+02, confidence 2.165E-01, row_norm 9.413E-01}
[06/14/2024 07:15:08 gluefactory INFO] [E 11 | it 1200] loss {total 6.209E-01, last 1.525E-01, assignment_nll 1.525E-01, nll_pos 1.862E-01, nll_neg 1.188E-01, num_matchable 1.373E+02, num_unmatchable 3.589E+02, confidence 1.962E-01, row_norm 9.649E-01}
[06/14/2024 07:18:31 gluefactory INFO] [E 11 | it 1300] loss {total 6.954E-01, last 1.943E-01, assignment_nll 1.943E-01, nll_pos 2.638E-01, nll_neg 1.248E-01, num_matchable 1.255E+02, num_unmatchable 3.710E+02, confidence 1.974E-01, row_norm 9.619E-01}
[06/14/2024 07:21:54 gluefactory INFO] [E 11 | it 1400] loss {total 7.514E-01, last 2.414E-01, assignment_nll 2.414E-01, nll_pos 3.574E-01, nll_neg 1.253E-01, num_matchable 1.346E+02, num_unmatchable 3.643E+02, confidence 1.916E-01, row_norm 9.526E-01}
[06/14/2024 07:25:17 gluefactory INFO] [E 11 | it 1500] loss {total 5.875E-01, last 1.344E-01, assignment_nll 1.344E-01, nll_pos 1.442E-01, nll_neg 1.246E-01, num_matchable 1.519E+02, num_unmatchable 3.454E+02, confidence 1.960E-01, row_norm 9.709E-01}
[06/14/2024 07:31:12 gluefactory INFO] [Validation] {match_recall 9.453E-01, match_precision 7.682E-01, accuracy 9.450E-01, average_precision 7.339E-01, loss/total 2.151E-01, loss/last 2.151E-01, loss/assignment_nll 2.151E-01, loss/nll_pos 2.773E-01, loss/nll_neg 1.529E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.597E-01}
[06/14/2024 07:38:39 gluefactory INFO] [Validation] {match_recall 9.461E-01, match_precision 7.806E-01, accuracy 9.495E-01, average_precision 7.463E-01, loss/total 2.119E-01, loss/last 2.119E-01, loss/assignment_nll 2.119E-01, loss/nll_pos 2.899E-01, loss/nll_neg 1.339E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.590E-01}
[06/14/2024 07:38:39 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 07:38:39 gluefactory INFO] New best val: loss/total=0.21193359062662598
[06/14/2024 07:38:41 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_11_18539.tar
[06/14/2024 07:38:41 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_3_6179.tar
[06/14/2024 07:38:42 gluefactory INFO] Starting epoch 12
[06/14/2024 07:38:42 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 07:38:47 gluefactory INFO] [E 12 | it 0] loss {total 7.475E-01, last 2.220E-01, assignment_nll 2.220E-01, nll_pos 3.037E-01, nll_neg 1.404E-01, num_matchable 1.261E+02, num_unmatchable 3.720E+02, confidence 2.003E-01, row_norm 9.596E-01}
[06/14/2024 07:42:10 gluefactory INFO] [E 12 | it 100] loss {total 6.779E-01, last 1.649E-01, assignment_nll 1.649E-01, nll_pos 1.852E-01, nll_neg 1.447E-01, num_matchable 1.316E+02, num_unmatchable 3.645E+02, confidence 2.019E-01, row_norm 9.580E-01}
[06/14/2024 07:45:33 gluefactory INFO] [E 12 | it 200] loss {total 6.428E-01, last 1.375E-01, assignment_nll 1.375E-01, nll_pos 1.555E-01, nll_neg 1.196E-01, num_matchable 1.239E+02, num_unmatchable 3.710E+02, confidence 1.993E-01, row_norm 9.665E-01}
[06/14/2024 07:48:56 gluefactory INFO] [E 12 | it 300] loss {total 7.180E-01, last 2.053E-01, assignment_nll 2.053E-01, nll_pos 2.524E-01, nll_neg 1.582E-01, num_matchable 1.127E+02, num_unmatchable 3.851E+02, confidence 1.890E-01, row_norm 9.518E-01}
[06/14/2024 07:52:19 gluefactory INFO] [E 12 | it 400] loss {total 6.419E-01, last 1.505E-01, assignment_nll 1.505E-01, nll_pos 1.523E-01, nll_neg 1.486E-01, num_matchable 1.122E+02, num_unmatchable 3.862E+02, confidence 1.889E-01, row_norm 9.697E-01}
[06/14/2024 07:55:41 gluefactory INFO] [E 12 | it 500] loss {total 7.464E-01, last 2.838E-01, assignment_nll 2.838E-01, nll_pos 4.251E-01, nll_neg 1.426E-01, num_matchable 1.212E+02, num_unmatchable 3.775E+02, confidence 1.937E-01, row_norm 9.563E-01}
[06/14/2024 08:01:35 gluefactory INFO] [Validation] {match_recall 9.477E-01, match_precision 7.879E-01, accuracy 9.520E-01, average_precision 7.536E-01, loss/total 2.029E-01, loss/last 2.029E-01, loss/assignment_nll 2.029E-01, loss/nll_pos 2.781E-01, loss/nll_neg 1.277E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.623E-01}
[06/14/2024 08:01:35 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 08:01:35 gluefactory INFO] New best val: loss/total=0.20292055197039016
[06/14/2024 08:04:58 gluefactory INFO] [E 12 | it 600] loss {total 6.192E-01, last 1.327E-01, assignment_nll 1.327E-01, nll_pos 1.542E-01, nll_neg 1.112E-01, num_matchable 1.214E+02, num_unmatchable 3.739E+02, confidence 1.983E-01, row_norm 9.728E-01}
[06/14/2024 08:08:20 gluefactory INFO] [E 12 | it 700] loss {total 7.097E-01, last 1.986E-01, assignment_nll 1.986E-01, nll_pos 2.676E-01, nll_neg 1.297E-01, num_matchable 1.187E+02, num_unmatchable 3.796E+02, confidence 1.950E-01, row_norm 9.608E-01}
[06/14/2024 08:11:43 gluefactory INFO] [E 12 | it 800] loss {total 7.175E-01, last 2.159E-01, assignment_nll 2.159E-01, nll_pos 2.916E-01, nll_neg 1.402E-01, num_matchable 1.186E+02, num_unmatchable 3.789E+02, confidence 1.930E-01, row_norm 9.598E-01}
[06/14/2024 08:15:05 gluefactory INFO] [E 12 | it 900] loss {total 6.688E-01, last 1.817E-01, assignment_nll 1.817E-01, nll_pos 2.176E-01, nll_neg 1.459E-01, num_matchable 1.253E+02, num_unmatchable 3.726E+02, confidence 1.985E-01, row_norm 9.652E-01}
[06/14/2024 08:18:28 gluefactory INFO] [E 12 | it 1000] loss {total 5.874E-01, last 1.358E-01, assignment_nll 1.358E-01, nll_pos 1.631E-01, nll_neg 1.085E-01, num_matchable 1.416E+02, num_unmatchable 3.554E+02, confidence 1.857E-01, row_norm 9.704E-01}
[06/14/2024 08:24:24 gluefactory INFO] [Validation] {match_recall 9.494E-01, match_precision 7.838E-01, accuracy 9.507E-01, average_precision 7.508E-01, loss/total 1.979E-01, loss/last 1.979E-01, loss/assignment_nll 1.979E-01, loss/nll_pos 2.600E-01, loss/nll_neg 1.358E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.609E-01}
[06/14/2024 08:24:24 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 08:24:24 gluefactory INFO] New best val: loss/total=0.19787437642150005
[06/14/2024 08:27:49 gluefactory INFO] [E 12 | it 1100] loss {total 6.542E-01, last 1.497E-01, assignment_nll 1.497E-01, nll_pos 1.761E-01, nll_neg 1.233E-01, num_matchable 1.348E+02, num_unmatchable 3.643E+02, confidence 2.027E-01, row_norm 9.711E-01}
[06/14/2024 08:31:12 gluefactory INFO] [E 12 | it 1200] loss {total 6.138E-01, last 1.355E-01, assignment_nll 1.355E-01, nll_pos 1.562E-01, nll_neg 1.149E-01, num_matchable 1.262E+02, num_unmatchable 3.697E+02, confidence 1.939E-01, row_norm 9.705E-01}
[06/14/2024 08:34:34 gluefactory INFO] [E 12 | it 1300] loss {total 7.407E-01, last 2.213E-01, assignment_nll 2.213E-01, nll_pos 3.074E-01, nll_neg 1.351E-01, num_matchable 1.254E+02, num_unmatchable 3.710E+02, confidence 2.056E-01, row_norm 9.596E-01}
[06/14/2024 08:37:57 gluefactory INFO] [E 12 | it 1400] loss {total 6.857E-01, last 2.169E-01, assignment_nll 2.169E-01, nll_pos 3.226E-01, nll_neg 1.113E-01, num_matchable 1.418E+02, num_unmatchable 3.574E+02, confidence 1.834E-01, row_norm 9.597E-01}
[06/14/2024 08:39:59 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_12_20000.tar
[06/14/2024 08:40:00 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_4_7724.tar
[06/14/2024 08:41:21 gluefactory INFO] [E 12 | it 1500] loss {total 5.913E-01, last 1.314E-01, assignment_nll 1.314E-01, nll_pos 1.566E-01, nll_neg 1.062E-01, num_matchable 1.363E+02, num_unmatchable 3.605E+02, confidence 1.908E-01, row_norm 9.665E-01}
[06/14/2024 08:47:16 gluefactory INFO] [Validation] {match_recall 9.447E-01, match_precision 7.781E-01, accuracy 9.488E-01, average_precision 7.436E-01, loss/total 2.178E-01, loss/last 2.178E-01, loss/assignment_nll 2.178E-01, loss/nll_pos 3.038E-01, loss/nll_neg 1.318E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.579E-01}
[06/14/2024 08:54:43 gluefactory INFO] [Validation] {match_recall 9.370E-01, match_precision 7.711E-01, accuracy 9.459E-01, average_precision 7.327E-01, loss/total 2.413E-01, loss/last 2.413E-01, loss/assignment_nll 2.413E-01, loss/nll_pos 3.432E-01, loss/nll_neg 1.394E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.532E-01}
[06/14/2024 08:54:44 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_12_20084.tar
[06/14/2024 08:54:45 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_5_9269.tar
[06/14/2024 08:54:45 gluefactory INFO] Starting epoch 13
[06/14/2024 08:54:45 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 08:54:50 gluefactory INFO] [E 13 | it 0] loss {total 8.173E-01, last 3.678E-01, assignment_nll 3.678E-01, nll_pos 5.815E-01, nll_neg 1.542E-01, num_matchable 1.249E+02, num_unmatchable 3.741E+02, confidence 2.040E-01, row_norm 9.500E-01}
[06/14/2024 08:58:13 gluefactory INFO] [E 13 | it 100] loss {total 6.285E-01, last 1.558E-01, assignment_nll 1.558E-01, nll_pos 1.807E-01, nll_neg 1.310E-01, num_matchable 1.421E+02, num_unmatchable 3.541E+02, confidence 1.985E-01, row_norm 9.662E-01}
[06/14/2024 09:01:36 gluefactory INFO] [E 13 | it 200] loss {total 6.610E-01, last 1.474E-01, assignment_nll 1.474E-01, nll_pos 1.768E-01, nll_neg 1.180E-01, num_matchable 1.308E+02, num_unmatchable 3.650E+02, confidence 1.958E-01, row_norm 9.637E-01}
[06/14/2024 09:04:58 gluefactory INFO] [E 13 | it 300] loss {total 6.472E-01, last 1.537E-01, assignment_nll 1.537E-01, nll_pos 1.915E-01, nll_neg 1.158E-01, num_matchable 1.149E+02, num_unmatchable 3.842E+02, confidence 1.874E-01, row_norm 9.673E-01}
[06/14/2024 09:08:20 gluefactory INFO] [E 13 | it 400] loss {total 7.824E-01, last 2.307E-01, assignment_nll 2.307E-01, nll_pos 2.900E-01, nll_neg 1.714E-01, num_matchable 1.110E+02, num_unmatchable 3.854E+02, confidence 2.048E-01, row_norm 9.478E-01}
[06/14/2024 09:11:41 gluefactory INFO] [E 13 | it 500] loss {total 8.674E-01, last 3.553E-01, assignment_nll 3.553E-01, nll_pos 5.401E-01, nll_neg 1.705E-01, num_matchable 1.232E+02, num_unmatchable 3.738E+02, confidence 1.992E-01, row_norm 9.523E-01}
[06/14/2024 09:17:35 gluefactory INFO] [Validation] {match_recall 9.499E-01, match_precision 7.891E-01, accuracy 9.528E-01, average_precision 7.560E-01, loss/total 1.931E-01, loss/last 1.931E-01, loss/assignment_nll 1.931E-01, loss/nll_pos 2.625E-01, loss/nll_neg 1.237E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.636E-01}
[06/14/2024 09:17:35 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 09:17:35 gluefactory INFO] New best val: loss/total=0.19309066277843148
[06/14/2024 09:20:59 gluefactory INFO] [E 13 | it 600] loss {total 6.104E-01, last 1.187E-01, assignment_nll 1.187E-01, nll_pos 1.164E-01, nll_neg 1.210E-01, num_matchable 1.270E+02, num_unmatchable 3.680E+02, confidence 2.031E-01, row_norm 9.741E-01}
[06/14/2024 09:24:22 gluefactory INFO] [E 13 | it 700] loss {total 5.971E-01, last 1.325E-01, assignment_nll 1.325E-01, nll_pos 1.461E-01, nll_neg 1.189E-01, num_matchable 1.369E+02, num_unmatchable 3.603E+02, confidence 1.885E-01, row_norm 9.716E-01}
[06/14/2024 09:27:45 gluefactory INFO] [E 13 | it 800] loss {total 7.827E-01, last 2.742E-01, assignment_nll 2.742E-01, nll_pos 4.119E-01, nll_neg 1.364E-01, num_matchable 1.110E+02, num_unmatchable 3.854E+02, confidence 2.032E-01, row_norm 9.516E-01}
[06/14/2024 09:31:08 gluefactory INFO] [E 13 | it 900] loss {total 5.764E-01, last 1.102E-01, assignment_nll 1.102E-01, nll_pos 1.144E-01, nll_neg 1.060E-01, num_matchable 1.267E+02, num_unmatchable 3.732E+02, confidence 1.879E-01, row_norm 9.724E-01}
[06/14/2024 09:34:31 gluefactory INFO] [E 13 | it 1000] loss {total 5.469E-01, last 1.199E-01, assignment_nll 1.199E-01, nll_pos 1.201E-01, nll_neg 1.197E-01, num_matchable 1.394E+02, num_unmatchable 3.558E+02, confidence 1.840E-01, row_norm 9.742E-01}
[06/14/2024 09:40:27 gluefactory INFO] [Validation] {match_recall 9.510E-01, match_precision 7.844E-01, accuracy 9.511E-01, average_precision 7.526E-01, loss/total 1.904E-01, loss/last 1.904E-01, loss/assignment_nll 1.904E-01, loss/nll_pos 2.427E-01, loss/nll_neg 1.382E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.595E-01}
[06/14/2024 09:40:27 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 09:40:28 gluefactory INFO] New best val: loss/total=0.19043329613866192
[06/14/2024 09:43:52 gluefactory INFO] [E 13 | it 1100] loss {total 6.831E-01, last 1.727E-01, assignment_nll 1.727E-01, nll_pos 2.194E-01, nll_neg 1.261E-01, num_matchable 1.377E+02, num_unmatchable 3.599E+02, confidence 2.082E-01, row_norm 9.610E-01}
[06/14/2024 09:47:15 gluefactory INFO] [E 13 | it 1200] loss {total 6.243E-01, last 1.486E-01, assignment_nll 1.486E-01, nll_pos 1.922E-01, nll_neg 1.050E-01, num_matchable 1.343E+02, num_unmatchable 3.624E+02, confidence 1.904E-01, row_norm 9.690E-01}
[06/14/2024 09:50:38 gluefactory INFO] [E 13 | it 1300] loss {total 6.637E-01, last 1.624E-01, assignment_nll 1.624E-01, nll_pos 1.917E-01, nll_neg 1.330E-01, num_matchable 1.293E+02, num_unmatchable 3.653E+02, confidence 1.994E-01, row_norm 9.613E-01}
[06/14/2024 09:54:01 gluefactory INFO] [E 13 | it 1400] loss {total 7.026E-01, last 1.739E-01, assignment_nll 1.739E-01, nll_pos 2.253E-01, nll_neg 1.225E-01, num_matchable 1.404E+02, num_unmatchable 3.571E+02, confidence 2.019E-01, row_norm 9.637E-01}
[06/14/2024 09:57:24 gluefactory INFO] [E 13 | it 1500] loss {total 5.384E-01, last 1.008E-01, assignment_nll 1.008E-01, nll_pos 1.036E-01, nll_neg 9.809E-02, num_matchable 1.430E+02, num_unmatchable 3.530E+02, confidence 1.917E-01, row_norm 9.742E-01}
[06/14/2024 10:03:19 gluefactory INFO] [Validation] {match_recall 9.513E-01, match_precision 7.892E-01, accuracy 9.526E-01, average_precision 7.567E-01, loss/total 1.860E-01, loss/last 1.860E-01, loss/assignment_nll 1.860E-01, loss/nll_pos 2.504E-01, loss/nll_neg 1.216E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.630E-01}
[06/14/2024 10:03:19 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 10:03:20 gluefactory INFO] New best val: loss/total=0.18598787107485393
[06/14/2024 10:10:47 gluefactory INFO] [Validation] {match_recall 9.528E-01, match_precision 7.895E-01, accuracy 9.529E-01, average_precision 7.580E-01, loss/total 1.847E-01, loss/last 1.847E-01, loss/assignment_nll 1.847E-01, loss/nll_pos 2.493E-01, loss/nll_neg 1.201E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.621E-01}
[06/14/2024 10:10:47 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 10:10:48 gluefactory INFO] New best val: loss/total=0.1847085808122427
[06/14/2024 10:10:49 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_13_21629.tar
[06/14/2024 10:10:50 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_6_10000.tar
[06/14/2024 10:10:50 gluefactory INFO] Starting epoch 14
[06/14/2024 10:10:50 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 10:10:55 gluefactory INFO] [E 14 | it 0] loss {total 6.046E-01, last 1.252E-01, assignment_nll 1.252E-01, nll_pos 1.389E-01, nll_neg 1.114E-01, num_matchable 1.312E+02, num_unmatchable 3.682E+02, confidence 1.877E-01, row_norm 9.732E-01}
[06/14/2024 10:14:17 gluefactory INFO] [E 14 | it 100] loss {total 6.382E-01, last 1.594E-01, assignment_nll 1.594E-01, nll_pos 1.905E-01, nll_neg 1.282E-01, num_matchable 1.243E+02, num_unmatchable 3.752E+02, confidence 1.861E-01, row_norm 9.643E-01}
[06/14/2024 10:17:39 gluefactory INFO] [E 14 | it 200] loss {total 5.406E-01, last 1.012E-01, assignment_nll 1.012E-01, nll_pos 1.056E-01, nll_neg 9.677E-02, num_matchable 1.279E+02, num_unmatchable 3.678E+02, confidence 1.776E-01, row_norm 9.760E-01}
[06/14/2024 10:21:00 gluefactory INFO] [E 14 | it 300] loss {total 6.555E-01, last 1.710E-01, assignment_nll 1.710E-01, nll_pos 2.271E-01, nll_neg 1.149E-01, num_matchable 1.254E+02, num_unmatchable 3.726E+02, confidence 1.897E-01, row_norm 9.676E-01}
[06/14/2024 10:24:22 gluefactory INFO] [E 14 | it 400] loss {total 6.458E-01, last 1.643E-01, assignment_nll 1.643E-01, nll_pos 1.949E-01, nll_neg 1.337E-01, num_matchable 1.090E+02, num_unmatchable 3.882E+02, confidence 1.877E-01, row_norm 9.597E-01}
[06/14/2024 10:27:45 gluefactory INFO] [E 14 | it 500] loss {total 6.234E-01, last 1.213E-01, assignment_nll 1.213E-01, nll_pos 1.214E-01, nll_neg 1.213E-01, num_matchable 1.290E+02, num_unmatchable 3.704E+02, confidence 1.966E-01, row_norm 9.696E-01}
[06/14/2024 10:33:40 gluefactory INFO] [Validation] {match_recall 9.481E-01, match_precision 7.904E-01, accuracy 9.531E-01, average_precision 7.567E-01, loss/total 1.988E-01, loss/last 1.988E-01, loss/assignment_nll 1.988E-01, loss/nll_pos 2.780E-01, loss/nll_neg 1.195E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.597E-01}
[06/14/2024 10:37:04 gluefactory INFO] [E 14 | it 600] loss {total 6.950E-01, last 2.291E-01, assignment_nll 2.291E-01, nll_pos 3.216E-01, nll_neg 1.366E-01, num_matchable 1.210E+02, num_unmatchable 3.732E+02, confidence 1.868E-01, row_norm 9.578E-01}
[06/14/2024 10:40:27 gluefactory INFO] [E 14 | it 700] loss {total 5.855E-01, last 1.234E-01, assignment_nll 1.234E-01, nll_pos 1.362E-01, nll_neg 1.105E-01, num_matchable 1.200E+02, num_unmatchable 3.797E+02, confidence 1.787E-01, row_norm 9.720E-01}
[06/14/2024 10:43:50 gluefactory INFO] [E 14 | it 800] loss {total 7.739E-01, last 2.613E-01, assignment_nll 2.613E-01, nll_pos 3.493E-01, nll_neg 1.733E-01, num_matchable 1.157E+02, num_unmatchable 3.814E+02, confidence 1.976E-01, row_norm 9.504E-01}
[06/14/2024 10:47:13 gluefactory INFO] [E 14 | it 900] loss {total 5.366E-01, last 9.995E-02, assignment_nll 9.995E-02, nll_pos 9.456E-02, nll_neg 1.053E-01, num_matchable 1.198E+02, num_unmatchable 3.793E+02, confidence 1.813E-01, row_norm 9.753E-01}
[06/14/2024 10:50:36 gluefactory INFO] [E 14 | it 1000] loss {total 5.816E-01, last 1.149E-01, assignment_nll 1.149E-01, nll_pos 1.274E-01, nll_neg 1.024E-01, num_matchable 1.445E+02, num_unmatchable 3.503E+02, confidence 2.000E-01, row_norm 9.714E-01}
[06/14/2024 10:56:31 gluefactory INFO] [Validation] {match_recall 9.520E-01, match_precision 7.907E-01, accuracy 9.533E-01, average_precision 7.587E-01, loss/total 1.874E-01, loss/last 1.874E-01, loss/assignment_nll 1.874E-01, loss/nll_pos 2.581E-01, loss/nll_neg 1.167E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.614E-01}
[06/14/2024 10:59:56 gluefactory INFO] [E 14 | it 1100] loss {total 5.191E-01, last 9.296E-02, assignment_nll 9.296E-02, nll_pos 7.313E-02, nll_neg 1.128E-01, num_matchable 1.375E+02, num_unmatchable 3.620E+02, confidence 1.777E-01, row_norm 9.780E-01}
[06/14/2024 11:03:19 gluefactory INFO] [E 14 | it 1200] loss {total 6.516E-01, last 1.460E-01, assignment_nll 1.460E-01, nll_pos 1.725E-01, nll_neg 1.195E-01, num_matchable 1.249E+02, num_unmatchable 3.698E+02, confidence 1.944E-01, row_norm 9.657E-01}
[06/14/2024 11:06:42 gluefactory INFO] [E 14 | it 1300] loss {total 6.762E-01, last 1.706E-01, assignment_nll 1.706E-01, nll_pos 2.096E-01, nll_neg 1.317E-01, num_matchable 1.328E+02, num_unmatchable 3.630E+02, confidence 1.979E-01, row_norm 9.606E-01}
[06/14/2024 11:10:05 gluefactory INFO] [E 14 | it 1400] loss {total 6.760E-01, last 1.730E-01, assignment_nll 1.730E-01, nll_pos 2.268E-01, nll_neg 1.192E-01, num_matchable 1.339E+02, num_unmatchable 3.642E+02, confidence 2.016E-01, row_norm 9.632E-01}
[06/14/2024 11:13:28 gluefactory INFO] [E 14 | it 1500] loss {total 8.781E-01, last 4.138E-01, assignment_nll 4.138E-01, nll_pos 6.441E-01, nll_neg 1.834E-01, num_matchable 1.307E+02, num_unmatchable 3.640E+02, confidence 1.995E-01, row_norm 9.239E-01}
[06/14/2024 11:19:23 gluefactory INFO] [Validation] {match_recall 9.456E-01, match_precision 7.816E-01, accuracy 9.502E-01, average_precision 7.471E-01, loss/total 2.099E-01, loss/last 2.099E-01, loss/assignment_nll 2.099E-01, loss/nll_pos 2.805E-01, loss/nll_neg 1.393E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.546E-01}
[06/14/2024 11:26:48 gluefactory INFO] [Validation] {match_recall 9.453E-01, match_precision 7.748E-01, accuracy 9.474E-01, average_precision 7.403E-01, loss/total 2.161E-01, loss/last 2.161E-01, loss/assignment_nll 2.161E-01, loss/nll_pos 2.848E-01, loss/nll_neg 1.475E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.592E-01}
[06/14/2024 11:26:50 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_14_23174.tar
[06/14/2024 11:26:50 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_6_10814.tar
[06/14/2024 11:26:50 gluefactory INFO] Starting epoch 15
[06/14/2024 11:26:50 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 11:26:56 gluefactory INFO] [E 15 | it 0] loss {total 7.444E-01, last 2.287E-01, assignment_nll 2.287E-01, nll_pos 3.113E-01, nll_neg 1.461E-01, num_matchable 1.229E+02, num_unmatchable 3.759E+02, confidence 2.008E-01, row_norm 9.668E-01}
[06/14/2024 11:30:17 gluefactory INFO] [E 15 | it 100] loss {total 6.644E-01, last 1.598E-01, assignment_nll 1.598E-01, nll_pos 1.945E-01, nll_neg 1.252E-01, num_matchable 1.282E+02, num_unmatchable 3.711E+02, confidence 1.947E-01, row_norm 9.672E-01}
[06/14/2024 11:33:39 gluefactory INFO] [E 15 | it 200] loss {total 6.248E-01, last 1.207E-01, assignment_nll 1.207E-01, nll_pos 1.413E-01, nll_neg 1.001E-01, num_matchable 1.390E+02, num_unmatchable 3.532E+02, confidence 1.974E-01, row_norm 9.666E-01}
[06/14/2024 11:37:02 gluefactory INFO] [E 15 | it 300] loss {total 7.098E-01, last 1.643E-01, assignment_nll 1.643E-01, nll_pos 1.932E-01, nll_neg 1.353E-01, num_matchable 1.118E+02, num_unmatchable 3.859E+02, confidence 1.943E-01, row_norm 9.664E-01}
[06/14/2024 11:40:24 gluefactory INFO] [E 15 | it 400] loss {total 6.852E-01, last 2.189E-01, assignment_nll 2.189E-01, nll_pos 3.043E-01, nll_neg 1.335E-01, num_matchable 1.132E+02, num_unmatchable 3.851E+02, confidence 1.769E-01, row_norm 9.526E-01}
[06/14/2024 11:43:47 gluefactory INFO] [E 15 | it 500] loss {total 8.358E-01, last 3.616E-01, assignment_nll 3.616E-01, nll_pos 5.798E-01, nll_neg 1.434E-01, num_matchable 1.181E+02, num_unmatchable 3.812E+02, confidence 1.861E-01, row_norm 9.522E-01}
[06/14/2024 11:49:42 gluefactory INFO] [Validation] {match_recall 9.529E-01, match_precision 7.959E-01, accuracy 9.550E-01, average_precision 7.641E-01, loss/total 1.804E-01, loss/last 1.804E-01, loss/assignment_nll 1.804E-01, loss/nll_pos 2.449E-01, loss/nll_neg 1.159E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.644E-01}
[06/14/2024 11:49:42 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 11:49:43 gluefactory INFO] New best val: loss/total=0.18038359161218973
[06/14/2024 11:53:07 gluefactory INFO] [E 15 | it 600] loss {total 5.763E-01, last 1.142E-01, assignment_nll 1.142E-01, nll_pos 1.152E-01, nll_neg 1.132E-01, num_matchable 1.247E+02, num_unmatchable 3.702E+02, confidence 1.895E-01, row_norm 9.733E-01}
[06/14/2024 11:56:30 gluefactory INFO] [E 15 | it 700] loss {total 6.093E-01, last 1.371E-01, assignment_nll 1.371E-01, nll_pos 1.771E-01, nll_neg 9.711E-02, num_matchable 1.288E+02, num_unmatchable 3.685E+02, confidence 1.892E-01, row_norm 9.724E-01}
[06/14/2024 11:59:52 gluefactory INFO] [E 15 | it 800] loss {total 7.064E-01, last 2.251E-01, assignment_nll 2.251E-01, nll_pos 3.143E-01, nll_neg 1.359E-01, num_matchable 1.088E+02, num_unmatchable 3.888E+02, confidence 1.846E-01, row_norm 9.636E-01}
[06/14/2024 12:03:15 gluefactory INFO] [E 15 | it 900] loss {total 7.737E-01, last 3.115E-01, assignment_nll 3.115E-01, nll_pos 5.010E-01, nll_neg 1.220E-01, num_matchable 1.257E+02, num_unmatchable 3.732E+02, confidence 1.770E-01, row_norm 9.523E-01}
[06/14/2024 12:06:38 gluefactory INFO] [E 15 | it 1000] loss {total 5.867E-01, last 1.155E-01, assignment_nll 1.155E-01, nll_pos 1.269E-01, nll_neg 1.040E-01, num_matchable 1.380E+02, num_unmatchable 3.567E+02, confidence 1.945E-01, row_norm 9.676E-01}
[06/14/2024 12:12:33 gluefactory INFO] [Validation] {match_recall 9.540E-01, match_precision 7.942E-01, accuracy 9.546E-01, average_precision 7.635E-01, loss/total 1.795E-01, loss/last 1.795E-01, loss/assignment_nll 1.795E-01, loss/nll_pos 2.397E-01, loss/nll_neg 1.193E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.650E-01}
[06/14/2024 12:12:33 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 12:12:33 gluefactory INFO] New best val: loss/total=0.17951547126597228
[06/14/2024 12:15:58 gluefactory INFO] [E 15 | it 1100] loss {total 5.809E-01, last 1.320E-01, assignment_nll 1.320E-01, nll_pos 1.572E-01, nll_neg 1.068E-01, num_matchable 1.326E+02, num_unmatchable 3.651E+02, confidence 1.855E-01, row_norm 9.747E-01}
[06/14/2024 12:19:20 gluefactory INFO] [E 15 | it 1200] loss {total 5.931E-01, last 1.189E-01, assignment_nll 1.189E-01, nll_pos 1.447E-01, nll_neg 9.316E-02, num_matchable 1.281E+02, num_unmatchable 3.667E+02, confidence 1.882E-01, row_norm 9.739E-01}
[06/14/2024 12:22:43 gluefactory INFO] [E 15 | it 1300] loss {total 6.111E-01, last 1.336E-01, assignment_nll 1.336E-01, nll_pos 1.606E-01, nll_neg 1.066E-01, num_matchable 1.387E+02, num_unmatchable 3.590E+02, confidence 1.910E-01, row_norm 9.646E-01}
[06/14/2024 12:26:06 gluefactory INFO] [E 15 | it 1400] loss {total 5.679E-01, last 1.166E-01, assignment_nll 1.166E-01, nll_pos 1.275E-01, nll_neg 1.057E-01, num_matchable 1.337E+02, num_unmatchable 3.657E+02, confidence 1.838E-01, row_norm 9.761E-01}
[06/14/2024 12:29:29 gluefactory INFO] [E 15 | it 1500] loss {total 6.768E-01, last 2.061E-01, assignment_nll 2.061E-01, nll_pos 2.458E-01, nll_neg 1.665E-01, num_matchable 1.429E+02, num_unmatchable 3.533E+02, confidence 2.077E-01, row_norm 9.461E-01}
[06/14/2024 12:35:21 gluefactory INFO] [Validation] {match_recall 9.358E-01, match_precision 7.553E-01, accuracy 9.400E-01, average_precision 7.177E-01, loss/total 2.612E-01, loss/last 2.612E-01, loss/assignment_nll 2.612E-01, loss/nll_pos 3.499E-01, loss/nll_neg 1.725E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.501E-01}
[06/14/2024 12:42:46 gluefactory INFO] [Validation] {match_recall 9.474E-01, match_precision 7.761E-01, accuracy 9.480E-01, average_precision 7.436E-01, loss/total 2.081E-01, loss/last 2.081E-01, loss/assignment_nll 2.081E-01, loss/nll_pos 2.767E-01, loss/nll_neg 1.396E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.558E-01}
[06/14/2024 12:42:47 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_15_24719.tar
[06/14/2024 12:42:48 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_7_12359.tar
[06/14/2024 12:42:48 gluefactory INFO] Starting epoch 16
[06/14/2024 12:42:48 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 12:42:54 gluefactory INFO] [E 16 | it 0] loss {total 6.715E-01, last 1.859E-01, assignment_nll 1.859E-01, nll_pos 2.362E-01, nll_neg 1.356E-01, num_matchable 1.203E+02, num_unmatchable 3.777E+02, confidence 1.980E-01, row_norm 9.626E-01}
[06/14/2024 12:46:17 gluefactory INFO] [E 16 | it 100] loss {total 6.548E-01, last 1.486E-01, assignment_nll 1.486E-01, nll_pos 1.907E-01, nll_neg 1.065E-01, num_matchable 1.295E+02, num_unmatchable 3.687E+02, confidence 2.025E-01, row_norm 9.660E-01}
[06/14/2024 12:49:39 gluefactory INFO] [E 16 | it 200] loss {total 6.346E-01, last 1.392E-01, assignment_nll 1.392E-01, nll_pos 1.710E-01, nll_neg 1.074E-01, num_matchable 1.264E+02, num_unmatchable 3.694E+02, confidence 1.853E-01, row_norm 9.725E-01}
[06/14/2024 12:52:21 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_16_25000.tar
[06/14/2024 12:52:22 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_8_13904.tar
[06/14/2024 12:53:03 gluefactory INFO] [E 16 | it 300] loss {total 7.540E-01, last 3.214E-01, assignment_nll 3.214E-01, nll_pos 5.280E-01, nll_neg 1.147E-01, num_matchable 1.257E+02, num_unmatchable 3.720E+02, confidence 1.820E-01, row_norm 9.681E-01}
[06/14/2024 12:56:25 gluefactory INFO] [E 16 | it 400] loss {total 7.628E-01, last 2.533E-01, assignment_nll 2.533E-01, nll_pos 3.661E-01, nll_neg 1.406E-01, num_matchable 1.203E+02, num_unmatchable 3.758E+02, confidence 1.936E-01, row_norm 9.560E-01}
[06/14/2024 12:59:48 gluefactory INFO] [E 16 | it 500] loss {total 5.873E-01, last 1.246E-01, assignment_nll 1.246E-01, nll_pos 1.400E-01, nll_neg 1.091E-01, num_matchable 1.288E+02, num_unmatchable 3.704E+02, confidence 1.891E-01, row_norm 9.726E-01}
[06/14/2024 13:05:43 gluefactory INFO] [Validation] {match_recall 9.569E-01, match_precision 8.034E-01, accuracy 9.576E-01, average_precision 7.734E-01, loss/total 1.673E-01, loss/last 1.673E-01, loss/assignment_nll 1.673E-01, loss/nll_pos 2.227E-01, loss/nll_neg 1.118E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.665E-01}
[06/14/2024 13:05:43 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 13:05:44 gluefactory INFO] New best val: loss/total=0.1672680410744447
[06/14/2024 13:09:08 gluefactory INFO] [E 16 | it 600] loss {total 6.180E-01, last 1.347E-01, assignment_nll 1.347E-01, nll_pos 1.660E-01, nll_neg 1.034E-01, num_matchable 1.254E+02, num_unmatchable 3.692E+02, confidence 1.863E-01, row_norm 9.741E-01}
[06/14/2024 13:12:31 gluefactory INFO] [E 16 | it 700] loss {total 5.052E-01, last 8.534E-02, assignment_nll 8.534E-02, nll_pos 7.282E-02, nll_neg 9.785E-02, num_matchable 1.296E+02, num_unmatchable 3.695E+02, confidence 1.669E-01, row_norm 9.785E-01}
[06/14/2024 13:15:54 gluefactory INFO] [E 16 | it 800] loss {total 6.146E-01, last 1.317E-01, assignment_nll 1.317E-01, nll_pos 1.551E-01, nll_neg 1.082E-01, num_matchable 1.198E+02, num_unmatchable 3.766E+02, confidence 1.872E-01, row_norm 9.687E-01}
[06/14/2024 13:19:17 gluefactory INFO] [E 16 | it 900] loss {total 5.319E-01, last 9.664E-02, assignment_nll 9.664E-02, nll_pos 9.445E-02, nll_neg 9.883E-02, num_matchable 1.298E+02, num_unmatchable 3.682E+02, confidence 1.857E-01, row_norm 9.746E-01}
[06/14/2024 13:22:40 gluefactory INFO] [E 16 | it 1000] loss {total 5.614E-01, last 1.469E-01, assignment_nll 1.469E-01, nll_pos 1.793E-01, nll_neg 1.145E-01, num_matchable 1.306E+02, num_unmatchable 3.667E+02, confidence 1.696E-01, row_norm 9.632E-01}
[06/14/2024 13:28:36 gluefactory INFO] [Validation] {match_recall 9.558E-01, match_precision 7.940E-01, accuracy 9.546E-01, average_precision 7.643E-01, loss/total 1.717E-01, loss/last 1.717E-01, loss/assignment_nll 1.717E-01, loss/nll_pos 2.144E-01, loss/nll_neg 1.290E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.627E-01}
[06/14/2024 13:32:01 gluefactory INFO] [E 16 | it 1100] loss {total 5.554E-01, last 1.084E-01, assignment_nll 1.084E-01, nll_pos 1.085E-01, nll_neg 1.082E-01, num_matchable 1.398E+02, num_unmatchable 3.582E+02, confidence 1.875E-01, row_norm 9.766E-01}
[06/14/2024 13:35:24 gluefactory INFO] [E 16 | it 1200] loss {total 6.237E-01, last 1.273E-01, assignment_nll 1.273E-01, nll_pos 1.427E-01, nll_neg 1.119E-01, num_matchable 1.276E+02, num_unmatchable 3.668E+02, confidence 1.964E-01, row_norm 9.738E-01}
[06/14/2024 13:38:46 gluefactory INFO] [E 16 | it 1300] loss {total 6.677E-01, last 1.345E-01, assignment_nll 1.345E-01, nll_pos 1.494E-01, nll_neg 1.196E-01, num_matchable 1.263E+02, num_unmatchable 3.711E+02, confidence 2.019E-01, row_norm 9.698E-01}
[06/14/2024 13:42:07 gluefactory INFO] [E 16 | it 1400] loss {total 6.431E-01, last 1.916E-01, assignment_nll 1.916E-01, nll_pos 2.843E-01, nll_neg 9.890E-02, num_matchable 1.410E+02, num_unmatchable 3.587E+02, confidence 1.776E-01, row_norm 9.698E-01}
[06/14/2024 13:45:28 gluefactory INFO] [E 16 | it 1500] loss {total 5.534E-01, last 1.076E-01, assignment_nll 1.076E-01, nll_pos 1.229E-01, nll_neg 9.220E-02, num_matchable 1.424E+02, num_unmatchable 3.542E+02, confidence 1.869E-01, row_norm 9.749E-01}
[06/14/2024 13:51:26 gluefactory INFO] [Validation] {match_recall 9.550E-01, match_precision 8.029E-01, accuracy 9.576E-01, average_precision 7.723E-01, loss/total 1.735E-01, loss/last 1.735E-01, loss/assignment_nll 1.735E-01, loss/nll_pos 2.370E-01, loss/nll_neg 1.101E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.680E-01}
[06/14/2024 13:58:54 gluefactory INFO] [Validation] {match_recall 9.538E-01, match_precision 7.949E-01, accuracy 9.549E-01, average_precision 7.640E-01, loss/total 1.791E-01, loss/last 1.791E-01, loss/assignment_nll 1.791E-01, loss/nll_pos 2.457E-01, loss/nll_neg 1.125E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.631E-01}
[06/14/2024 13:58:56 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_16_26264.tar
[06/14/2024 13:58:57 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_9_15000.tar
[06/14/2024 13:58:57 gluefactory INFO] Starting epoch 17
[06/14/2024 13:58:57 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 13:59:02 gluefactory INFO] [E 17 | it 0] loss {total 8.485E-01, last 4.200E-01, assignment_nll 4.200E-01, nll_pos 7.057E-01, nll_neg 1.343E-01, num_matchable 1.194E+02, num_unmatchable 3.798E+02, confidence 1.822E-01, row_norm 9.531E-01}
[06/14/2024 14:02:25 gluefactory INFO] [E 17 | it 100] loss {total 5.752E-01, last 1.269E-01, assignment_nll 1.269E-01, nll_pos 1.464E-01, nll_neg 1.074E-01, num_matchable 1.355E+02, num_unmatchable 3.620E+02, confidence 1.819E-01, row_norm 9.705E-01}
[06/14/2024 14:05:48 gluefactory INFO] [E 17 | it 200] loss {total 6.240E-01, last 1.425E-01, assignment_nll 1.425E-01, nll_pos 1.734E-01, nll_neg 1.116E-01, num_matchable 1.345E+02, num_unmatchable 3.610E+02, confidence 1.794E-01, row_norm 9.647E-01}
[06/14/2024 14:09:12 gluefactory INFO] [E 17 | it 300] loss {total 6.159E-01, last 1.372E-01, assignment_nll 1.372E-01, nll_pos 1.700E-01, nll_neg 1.044E-01, num_matchable 1.286E+02, num_unmatchable 3.694E+02, confidence 1.866E-01, row_norm 9.704E-01}
[06/14/2024 14:12:35 gluefactory INFO] [E 17 | it 400] loss {total 6.155E-01, last 1.386E-01, assignment_nll 1.386E-01, nll_pos 1.707E-01, nll_neg 1.064E-01, num_matchable 1.127E+02, num_unmatchable 3.851E+02, confidence 1.795E-01, row_norm 9.697E-01}
[06/14/2024 14:15:58 gluefactory INFO] [E 17 | it 500] loss {total 5.558E-01, last 1.079E-01, assignment_nll 1.079E-01, nll_pos 1.252E-01, nll_neg 9.055E-02, num_matchable 1.326E+02, num_unmatchable 3.657E+02, confidence 1.833E-01, row_norm 9.783E-01}
[06/14/2024 14:21:54 gluefactory INFO] [Validation] {match_recall 9.573E-01, match_precision 8.128E-01, accuracy 9.609E-01, average_precision 7.825E-01, loss/total 1.617E-01, loss/last 1.617E-01, loss/assignment_nll 1.617E-01, loss/nll_pos 2.254E-01, loss/nll_neg 9.793E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.692E-01}
[06/14/2024 14:21:54 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 14:21:55 gluefactory INFO] New best val: loss/total=0.16166555502552976
[06/14/2024 14:25:19 gluefactory INFO] [E 17 | it 600] loss {total 5.286E-01, last 1.145E-01, assignment_nll 1.145E-01, nll_pos 1.347E-01, nll_neg 9.431E-02, num_matchable 1.210E+02, num_unmatchable 3.769E+02, confidence 1.703E-01, row_norm 9.792E-01}
[06/14/2024 14:28:43 gluefactory INFO] [E 17 | it 700] loss {total 6.470E-01, last 1.495E-01, assignment_nll 1.495E-01, nll_pos 1.662E-01, nll_neg 1.329E-01, num_matchable 1.331E+02, num_unmatchable 3.651E+02, confidence 2.010E-01, row_norm 9.634E-01}
[06/14/2024 14:32:06 gluefactory INFO] [E 17 | it 800] loss {total 6.719E-01, last 1.528E-01, assignment_nll 1.528E-01, nll_pos 1.962E-01, nll_neg 1.094E-01, num_matchable 1.251E+02, num_unmatchable 3.699E+02, confidence 2.028E-01, row_norm 9.753E-01}
[06/14/2024 14:35:29 gluefactory INFO] [E 17 | it 900] loss {total 7.029E-01, last 1.819E-01, assignment_nll 1.819E-01, nll_pos 2.353E-01, nll_neg 1.286E-01, num_matchable 1.218E+02, num_unmatchable 3.751E+02, confidence 1.999E-01, row_norm 9.598E-01}
[06/14/2024 14:38:52 gluefactory INFO] [E 17 | it 1000] loss {total 6.850E-01, last 2.266E-01, assignment_nll 2.266E-01, nll_pos 3.418E-01, nll_neg 1.114E-01, num_matchable 1.354E+02, num_unmatchable 3.611E+02, confidence 1.877E-01, row_norm 9.574E-01}
[06/14/2024 14:44:48 gluefactory INFO] [Validation] {match_recall 9.548E-01, match_precision 8.009E-01, accuracy 9.570E-01, average_precision 7.705E-01, loss/total 1.748E-01, loss/last 1.748E-01, loss/assignment_nll 1.748E-01, loss/nll_pos 2.334E-01, loss/nll_neg 1.162E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.629E-01}
[06/14/2024 14:48:13 gluefactory INFO] [E 17 | it 1100] loss {total 5.137E-01, last 1.035E-01, assignment_nll 1.035E-01, nll_pos 1.174E-01, nll_neg 8.955E-02, num_matchable 1.281E+02, num_unmatchable 3.719E+02, confidence 1.688E-01, row_norm 9.786E-01}
[06/14/2024 14:51:35 gluefactory INFO] [E 17 | it 1200] loss {total 5.245E-01, last 9.582E-02, assignment_nll 9.582E-02, nll_pos 1.015E-01, nll_neg 9.015E-02, num_matchable 1.218E+02, num_unmatchable 3.759E+02, confidence 1.697E-01, row_norm 9.754E-01}
[06/14/2024 14:54:56 gluefactory INFO] [E 17 | it 1300] loss {total 6.514E-01, last 1.368E-01, assignment_nll 1.368E-01, nll_pos 1.769E-01, nll_neg 9.682E-02, num_matchable 1.359E+02, num_unmatchable 3.599E+02, confidence 2.022E-01, row_norm 9.660E-01}
[06/14/2024 14:58:18 gluefactory INFO] [E 17 | it 1400] loss {total 5.301E-01, last 9.357E-02, assignment_nll 9.357E-02, nll_pos 1.018E-01, nll_neg 8.533E-02, num_matchable 1.474E+02, num_unmatchable 3.508E+02, confidence 1.840E-01, row_norm 9.795E-01}
[06/14/2024 15:01:41 gluefactory INFO] [E 17 | it 1500] loss {total 5.448E-01, last 1.111E-01, assignment_nll 1.111E-01, nll_pos 1.358E-01, nll_neg 8.640E-02, num_matchable 1.433E+02, num_unmatchable 3.512E+02, confidence 1.869E-01, row_norm 9.737E-01}
[06/14/2024 15:07:37 gluefactory INFO] [Validation] {match_recall 9.563E-01, match_precision 8.109E-01, accuracy 9.602E-01, average_precision 7.805E-01, loss/total 1.676E-01, loss/last 1.676E-01, loss/assignment_nll 1.676E-01, loss/nll_pos 2.297E-01, loss/nll_neg 1.056E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.693E-01}
[06/14/2024 15:15:07 gluefactory INFO] [Validation] {match_recall 9.561E-01, match_precision 8.092E-01, accuracy 9.597E-01, average_precision 7.790E-01, loss/total 1.689E-01, loss/last 1.689E-01, loss/assignment_nll 1.689E-01, loss/nll_pos 2.327E-01, loss/nll_neg 1.051E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.660E-01}
[06/14/2024 15:15:08 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_17_27809.tar
[06/14/2024 15:15:09 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_9_15449.tar
[06/14/2024 15:15:09 gluefactory INFO] Starting epoch 18
[06/14/2024 15:15:09 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 15:15:14 gluefactory INFO] [E 18 | it 0] loss {total 6.327E-01, last 1.401E-01, assignment_nll 1.401E-01, nll_pos 1.735E-01, nll_neg 1.066E-01, num_matchable 1.283E+02, num_unmatchable 3.681E+02, confidence 1.935E-01, row_norm 9.731E-01}
[06/14/2024 15:18:37 gluefactory INFO] [E 18 | it 100] loss {total 5.882E-01, last 1.234E-01, assignment_nll 1.234E-01, nll_pos 1.365E-01, nll_neg 1.103E-01, num_matchable 1.370E+02, num_unmatchable 3.596E+02, confidence 1.859E-01, row_norm 9.697E-01}
[06/14/2024 15:22:00 gluefactory INFO] [E 18 | it 200] loss {total 5.473E-01, last 1.104E-01, assignment_nll 1.104E-01, nll_pos 1.194E-01, nll_neg 1.014E-01, num_matchable 1.330E+02, num_unmatchable 3.607E+02, confidence 1.769E-01, row_norm 9.747E-01}
[06/14/2024 15:25:24 gluefactory INFO] [E 18 | it 300] loss {total 6.234E-01, last 1.270E-01, assignment_nll 1.270E-01, nll_pos 1.677E-01, nll_neg 8.635E-02, num_matchable 1.191E+02, num_unmatchable 3.792E+02, confidence 1.808E-01, row_norm 9.722E-01}
[06/14/2024 15:28:47 gluefactory INFO] [E 18 | it 400] loss {total 6.258E-01, last 1.253E-01, assignment_nll 1.253E-01, nll_pos 1.584E-01, nll_neg 9.220E-02, num_matchable 1.224E+02, num_unmatchable 3.710E+02, confidence 1.911E-01, row_norm 9.646E-01}
[06/14/2024 15:32:10 gluefactory INFO] [E 18 | it 500] loss {total 5.775E-01, last 1.048E-01, assignment_nll 1.048E-01, nll_pos 1.147E-01, nll_neg 9.485E-02, num_matchable 1.284E+02, num_unmatchable 3.706E+02, confidence 1.858E-01, row_norm 9.775E-01}
[06/14/2024 15:38:07 gluefactory INFO] [Validation] {match_recall 9.533E-01, match_precision 8.057E-01, accuracy 9.587E-01, average_precision 7.742E-01, loss/total 1.792E-01, loss/last 1.792E-01, loss/assignment_nll 1.792E-01, loss/nll_pos 2.524E-01, loss/nll_neg 1.061E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.673E-01}
[06/14/2024 15:41:32 gluefactory INFO] [E 18 | it 600] loss {total 7.310E-01, last 2.840E-01, assignment_nll 2.840E-01, nll_pos 4.419E-01, nll_neg 1.260E-01, num_matchable 1.134E+02, num_unmatchable 3.809E+02, confidence 1.796E-01, row_norm 9.582E-01}
[06/14/2024 15:44:55 gluefactory INFO] [E 18 | it 700] loss {total 5.556E-01, last 1.095E-01, assignment_nll 1.095E-01, nll_pos 1.191E-01, nll_neg 9.997E-02, num_matchable 1.205E+02, num_unmatchable 3.778E+02, confidence 1.693E-01, row_norm 9.723E-01}
[06/14/2024 15:48:18 gluefactory INFO] [E 18 | it 800] loss {total 7.573E-01, last 3.034E-01, assignment_nll 3.034E-01, nll_pos 4.889E-01, nll_neg 1.179E-01, num_matchable 1.171E+02, num_unmatchable 3.808E+02, confidence 1.773E-01, row_norm 9.556E-01}
[06/14/2024 15:51:41 gluefactory INFO] [E 18 | it 900] loss {total 5.874E-01, last 1.310E-01, assignment_nll 1.310E-01, nll_pos 1.530E-01, nll_neg 1.089E-01, num_matchable 1.307E+02, num_unmatchable 3.660E+02, confidence 1.901E-01, row_norm 9.662E-01}
[06/14/2024 15:55:04 gluefactory INFO] [E 18 | it 1000] loss {total 5.345E-01, last 8.872E-02, assignment_nll 8.872E-02, nll_pos 9.008E-02, nll_neg 8.736E-02, num_matchable 1.445E+02, num_unmatchable 3.518E+02, confidence 1.877E-01, row_norm 9.770E-01}
[06/14/2024 16:00:59 gluefactory INFO] [Validation] {match_recall 9.574E-01, match_precision 8.065E-01, accuracy 9.587E-01, average_precision 7.771E-01, loss/total 1.625E-01, loss/last 1.625E-01, loss/assignment_nll 1.625E-01, loss/nll_pos 2.162E-01, loss/nll_neg 1.087E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.661E-01}
[06/14/2024 16:04:22 gluefactory INFO] [E 18 | it 1100] loss {total 6.427E-01, last 1.147E-01, assignment_nll 1.147E-01, nll_pos 1.266E-01, nll_neg 1.029E-01, num_matchable 1.218E+02, num_unmatchable 3.780E+02, confidence 1.972E-01, row_norm 9.739E-01}
[06/14/2024 16:07:44 gluefactory INFO] [E 18 | it 1200] loss {total 6.176E-01, last 1.745E-01, assignment_nll 1.745E-01, nll_pos 2.433E-01, nll_neg 1.057E-01, num_matchable 1.193E+02, num_unmatchable 3.786E+02, confidence 1.712E-01, row_norm 9.675E-01}
[06/14/2024 16:11:06 gluefactory INFO] [E 18 | it 1300] loss {total 6.904E-01, last 2.061E-01, assignment_nll 2.061E-01, nll_pos 3.013E-01, nll_neg 1.109E-01, num_matchable 1.255E+02, num_unmatchable 3.717E+02, confidence 1.865E-01, row_norm 9.601E-01}
[06/14/2024 16:14:30 gluefactory INFO] [E 18 | it 1400] loss {total 5.578E-01, last 1.290E-01, assignment_nll 1.290E-01, nll_pos 1.747E-01, nll_neg 8.335E-02, num_matchable 1.377E+02, num_unmatchable 3.616E+02, confidence 1.732E-01, row_norm 9.763E-01}
[06/14/2024 16:17:53 gluefactory INFO] [E 18 | it 1500] loss {total 5.845E-01, last 1.562E-01, assignment_nll 1.562E-01, nll_pos 2.129E-01, nll_neg 9.944E-02, num_matchable 1.467E+02, num_unmatchable 3.507E+02, confidence 1.809E-01, row_norm 9.622E-01}
[06/14/2024 16:23:49 gluefactory INFO] [Validation] {match_recall 9.569E-01, match_precision 8.115E-01, accuracy 9.604E-01, average_precision 7.814E-01, loss/total 1.635E-01, loss/last 1.635E-01, loss/assignment_nll 1.635E-01, loss/nll_pos 2.232E-01, loss/nll_neg 1.037E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.652E-01}
[06/14/2024 16:31:18 gluefactory INFO] [Validation] {match_recall 9.605E-01, match_precision 8.047E-01, accuracy 9.583E-01, average_precision 7.768E-01, loss/total 1.563E-01, loss/last 1.563E-01, loss/assignment_nll 1.563E-01, loss/nll_pos 2.027E-01, loss/nll_neg 1.098E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.690E-01}
[06/14/2024 16:31:18 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 16:31:18 gluefactory INFO] New best val: loss/total=0.15625700504756868
[06/14/2024 16:31:20 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_18_29354.tar
[06/14/2024 16:31:20 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_10_16994.tar
[06/14/2024 16:31:21 gluefactory INFO] Starting epoch 19
[06/14/2024 16:31:21 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/14/2024 16:31:26 gluefactory INFO] [E 19 | it 0] loss {total 5.940E-01, last 1.334E-01, assignment_nll 1.334E-01, nll_pos 1.540E-01, nll_neg 1.128E-01, num_matchable 1.179E+02, num_unmatchable 3.817E+02, confidence 1.749E-01, row_norm 9.750E-01}
[06/14/2024 16:34:49 gluefactory INFO] [E 19 | it 100] loss {total 6.051E-01, last 1.303E-01, assignment_nll 1.303E-01, nll_pos 1.705E-01, nll_neg 9.012E-02, num_matchable 1.189E+02, num_unmatchable 3.794E+02, confidence 1.867E-01, row_norm 9.720E-01}
[06/14/2024 16:38:12 gluefactory INFO] [E 19 | it 200] loss {total 5.734E-01, last 1.394E-01, assignment_nll 1.394E-01, nll_pos 1.740E-01, nll_neg 1.049E-01, num_matchable 1.228E+02, num_unmatchable 3.690E+02, confidence 1.784E-01, row_norm 9.719E-01}
[06/14/2024 16:41:36 gluefactory INFO] [E 19 | it 300] loss {total 6.411E-01, last 1.567E-01, assignment_nll 1.567E-01, nll_pos 1.791E-01, nll_neg 1.342E-01, num_matchable 1.172E+02, num_unmatchable 3.819E+02, confidence 1.891E-01, row_norm 9.625E-01}
[06/14/2024 16:44:59 gluefactory INFO] [E 19 | it 400] loss {total 6.121E-01, last 1.171E-01, assignment_nll 1.171E-01, nll_pos 1.266E-01, nll_neg 1.076E-01, num_matchable 1.123E+02, num_unmatchable 3.858E+02, confidence 1.766E-01, row_norm 9.713E-01}
[06/14/2024 16:48:22 gluefactory INFO] [E 19 | it 500] loss {total 6.423E-01, last 1.976E-01, assignment_nll 1.976E-01, nll_pos 2.714E-01, nll_neg 1.237E-01, num_matchable 1.326E+02, num_unmatchable 3.659E+02, confidence 1.678E-01, row_norm 9.637E-01}
[06/14/2024 16:54:18 gluefactory INFO] [Validation] {match_recall 9.598E-01, match_precision 8.074E-01, accuracy 9.591E-01, average_precision 7.788E-01, loss/total 1.554E-01, loss/last 1.554E-01, loss/assignment_nll 1.554E-01, loss/nll_pos 2.068E-01, loss/nll_neg 1.040E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.687E-01}
[06/14/2024 16:54:18 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 16:54:19 gluefactory INFO] New best val: loss/total=0.15538605299137137
[06/14/2024 16:57:43 gluefactory INFO] [E 19 | it 600] loss {total 5.185E-01, last 9.128E-02, assignment_nll 9.128E-02, nll_pos 9.395E-02, nll_neg 8.862E-02, num_matchable 1.274E+02, num_unmatchable 3.680E+02, confidence 1.758E-01, row_norm 9.807E-01}
[06/14/2024 16:59:14 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_19_30000.tar
[06/14/2024 16:59:15 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_11_18539.tar
[06/14/2024 17:01:07 gluefactory INFO] [E 19 | it 700] loss {total 5.259E-01, last 1.041E-01, assignment_nll 1.041E-01, nll_pos 1.199E-01, nll_neg 8.831E-02, num_matchable 1.258E+02, num_unmatchable 3.742E+02, confidence 1.732E-01, row_norm 9.781E-01}
[06/14/2024 17:04:30 gluefactory INFO] [E 19 | it 800] loss {total 6.064E-01, last 1.477E-01, assignment_nll 1.477E-01, nll_pos 1.983E-01, nll_neg 9.716E-02, num_matchable 1.168E+02, num_unmatchable 3.808E+02, confidence 1.804E-01, row_norm 9.730E-01}
[06/14/2024 17:07:52 gluefactory INFO] [E 19 | it 900] loss {total 5.388E-01, last 9.656E-02, assignment_nll 9.656E-02, nll_pos 1.030E-01, nll_neg 9.009E-02, num_matchable 1.187E+02, num_unmatchable 3.803E+02, confidence 1.761E-01, row_norm 9.747E-01}
[06/14/2024 17:11:14 gluefactory INFO] [E 19 | it 1000] loss {total 6.413E-01, last 2.181E-01, assignment_nll 2.181E-01, nll_pos 3.096E-01, nll_neg 1.266E-01, num_matchable 1.485E+02, num_unmatchable 3.475E+02, confidence 1.863E-01, row_norm 9.557E-01}
[06/14/2024 17:17:09 gluefactory INFO] [Validation] {match_recall 9.565E-01, match_precision 7.993E-01, accuracy 9.566E-01, average_precision 7.703E-01, loss/total 1.723E-01, loss/last 1.723E-01, loss/assignment_nll 1.723E-01, loss/nll_pos 2.233E-01, loss/nll_neg 1.214E-01, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.641E-01}
[06/14/2024 17:20:33 gluefactory INFO] [E 19 | it 1100] loss {total 5.748E-01, last 1.013E-01, assignment_nll 1.013E-01, nll_pos 1.152E-01, nll_neg 8.730E-02, num_matchable 1.372E+02, num_unmatchable 3.620E+02, confidence 1.889E-01, row_norm 9.796E-01}
[06/14/2024 17:23:56 gluefactory INFO] [E 19 | it 1200] loss {total 8.772E-01, last 4.336E-01, assignment_nll 4.336E-01, nll_pos 7.298E-01, nll_neg 1.374E-01, num_matchable 1.230E+02, num_unmatchable 3.731E+02, confidence 1.714E-01, row_norm 9.546E-01}
[06/14/2024 17:27:19 gluefactory INFO] [E 19 | it 1300] loss {total 5.518E-01, last 1.102E-01, assignment_nll 1.102E-01, nll_pos 1.262E-01, nll_neg 9.412E-02, num_matchable 1.281E+02, num_unmatchable 3.668E+02, confidence 1.862E-01, row_norm 9.703E-01}
[06/14/2024 17:30:43 gluefactory INFO] [E 19 | it 1400] loss {total 5.377E-01, last 9.391E-02, assignment_nll 9.391E-02, nll_pos 1.000E-01, nll_neg 8.777E-02, num_matchable 1.316E+02, num_unmatchable 3.674E+02, confidence 1.728E-01, row_norm 9.750E-01}
[06/14/2024 17:34:06 gluefactory INFO] [E 19 | it 1500] loss {total 5.524E-01, last 1.341E-01, assignment_nll 1.341E-01, nll_pos 1.812E-01, nll_neg 8.697E-02, num_matchable 1.426E+02, num_unmatchable 3.541E+02, confidence 1.771E-01, row_norm 9.747E-01}
[06/14/2024 17:40:02 gluefactory INFO] [Validation] {match_recall 9.619E-01, match_precision 8.178E-01, accuracy 9.625E-01, average_precision 7.898E-01, loss/total 1.461E-01, loss/last 1.461E-01, loss/assignment_nll 1.461E-01, loss/nll_pos 1.988E-01, loss/nll_neg 9.350E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.702E-01}
[06/14/2024 17:40:02 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 17:40:03 gluefactory INFO] New best val: loss/total=0.14613555901491998
[06/14/2024 17:47:31 gluefactory INFO] [Validation] {match_recall 9.549E-01, match_precision 8.291E-01, accuracy 9.655E-01, average_precision 7.967E-01, loss/total 1.639E-01, loss/last 1.639E-01, loss/assignment_nll 1.639E-01, loss/nll_pos 2.396E-01, loss/nll_neg 8.813E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.717E-01}
[06/14/2024 17:47:33 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_19_30899.tar
[06/14/2024 17:47:33 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_12_20000.tar
[06/14/2024 17:47:33 gluefactory INFO] Starting epoch 20
[06/14/2024 17:47:33 gluefactory INFO] lr changed from 0.0001 to 7.943282347242815e-05
[06/14/2024 17:47:39 gluefactory INFO] [E 20 | it 0] loss {total 6.938E-01, last 2.229E-01, assignment_nll 2.229E-01, nll_pos 3.505E-01, nll_neg 9.529E-02, num_matchable 1.224E+02, num_unmatchable 3.770E+02, confidence 1.836E-01, row_norm 9.712E-01}
[06/14/2024 17:51:02 gluefactory INFO] [E 20 | it 100] loss {total 5.145E-01, last 9.283E-02, assignment_nll 9.283E-02, nll_pos 9.816E-02, nll_neg 8.750E-02, num_matchable 1.255E+02, num_unmatchable 3.725E+02, confidence 1.746E-01, row_norm 9.758E-01}
[06/14/2024 17:54:25 gluefactory INFO] [E 20 | it 200] loss {total 5.032E-01, last 7.561E-02, assignment_nll 7.561E-02, nll_pos 7.785E-02, nll_neg 7.337E-02, num_matchable 1.250E+02, num_unmatchable 3.708E+02, confidence 1.709E-01, row_norm 9.787E-01}
[06/14/2024 17:57:48 gluefactory INFO] [E 20 | it 300] loss {total 4.974E-01, last 7.479E-02, assignment_nll 7.479E-02, nll_pos 7.302E-02, nll_neg 7.656E-02, num_matchable 1.232E+02, num_unmatchable 3.746E+02, confidence 1.709E-01, row_norm 9.805E-01}
[06/14/2024 18:01:11 gluefactory INFO] [E 20 | it 400] loss {total 5.896E-01, last 1.334E-01, assignment_nll 1.334E-01, nll_pos 1.548E-01, nll_neg 1.121E-01, num_matchable 1.110E+02, num_unmatchable 3.867E+02, confidence 1.719E-01, row_norm 9.706E-01}
[06/14/2024 18:04:34 gluefactory INFO] [E 20 | it 500] loss {total 5.538E-01, last 8.870E-02, assignment_nll 8.870E-02, nll_pos 8.926E-02, nll_neg 8.814E-02, num_matchable 1.316E+02, num_unmatchable 3.653E+02, confidence 1.901E-01, row_norm 9.797E-01}
[06/14/2024 18:10:31 gluefactory INFO] [Validation] {match_recall 9.599E-01, match_precision 8.228E-01, accuracy 9.639E-01, average_precision 7.938E-01, loss/total 1.502E-01, loss/last 1.502E-01, loss/assignment_nll 1.502E-01, loss/nll_pos 2.072E-01, loss/nll_neg 9.326E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.720E-01}
[06/14/2024 18:13:55 gluefactory INFO] [E 20 | it 600] loss {total 4.847E-01, last 8.018E-02, assignment_nll 8.018E-02, nll_pos 9.181E-02, nll_neg 6.854E-02, num_matchable 1.198E+02, num_unmatchable 3.765E+02, confidence 1.659E-01, row_norm 9.812E-01}
[06/14/2024 18:17:17 gluefactory INFO] [E 20 | it 700] loss {total 5.381E-01, last 1.037E-01, assignment_nll 1.037E-01, nll_pos 1.208E-01, nll_neg 8.654E-02, num_matchable 1.328E+02, num_unmatchable 3.641E+02, confidence 1.759E-01, row_norm 9.772E-01}
[06/14/2024 18:20:38 gluefactory INFO] [E 20 | it 800] loss {total 5.219E-01, last 8.963E-02, assignment_nll 8.963E-02, nll_pos 9.209E-02, nll_neg 8.717E-02, num_matchable 1.186E+02, num_unmatchable 3.790E+02, confidence 1.661E-01, row_norm 9.795E-01}
[06/14/2024 18:24:00 gluefactory INFO] [E 20 | it 900] loss {total 6.434E-01, last 2.113E-01, assignment_nll 2.113E-01, nll_pos 3.160E-01, nll_neg 1.066E-01, num_matchable 1.306E+02, num_unmatchable 3.671E+02, confidence 1.811E-01, row_norm 9.524E-01}
[06/14/2024 18:27:22 gluefactory INFO] [E 20 | it 1000] loss {total 5.118E-01, last 7.953E-02, assignment_nll 7.953E-02, nll_pos 7.280E-02, nll_neg 8.627E-02, num_matchable 1.432E+02, num_unmatchable 3.521E+02, confidence 1.797E-01, row_norm 9.800E-01}
[06/14/2024 18:33:18 gluefactory INFO] [Validation] {match_recall 9.595E-01, match_precision 8.250E-01, accuracy 9.648E-01, average_precision 7.962E-01, loss/total 1.490E-01, loss/last 1.490E-01, loss/assignment_nll 1.490E-01, loss/nll_pos 2.038E-01, loss/nll_neg 9.419E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.698E-01}
[06/14/2024 18:36:42 gluefactory INFO] [E 20 | it 1100] loss {total 5.001E-01, last 9.352E-02, assignment_nll 9.352E-02, nll_pos 1.195E-01, nll_neg 6.757E-02, num_matchable 1.272E+02, num_unmatchable 3.708E+02, confidence 1.676E-01, row_norm 9.777E-01}
[06/14/2024 18:40:05 gluefactory INFO] [E 20 | it 1200] loss {total 5.509E-01, last 9.382E-02, assignment_nll 9.382E-02, nll_pos 1.101E-01, nll_neg 7.752E-02, num_matchable 1.314E+02, num_unmatchable 3.618E+02, confidence 1.849E-01, row_norm 9.754E-01}
[06/14/2024 18:43:28 gluefactory INFO] [E 20 | it 1300] loss {total 5.370E-01, last 8.541E-02, assignment_nll 8.541E-02, nll_pos 9.317E-02, nll_neg 7.766E-02, num_matchable 1.341E+02, num_unmatchable 3.624E+02, confidence 1.824E-01, row_norm 9.778E-01}
[06/14/2024 18:46:51 gluefactory INFO] [E 20 | it 1400] loss {total 7.768E-01, last 3.524E-01, assignment_nll 3.524E-01, nll_pos 5.803E-01, nll_neg 1.245E-01, num_matchable 1.430E+02, num_unmatchable 3.558E+02, confidence 1.743E-01, row_norm 9.523E-01}
[06/14/2024 18:50:14 gluefactory INFO] [E 20 | it 1500] loss {total 5.039E-01, last 7.684E-02, assignment_nll 7.684E-02, nll_pos 7.631E-02, nll_neg 7.737E-02, num_matchable 1.356E+02, num_unmatchable 3.595E+02, confidence 1.819E-01, row_norm 9.727E-01}
[06/14/2024 18:56:09 gluefactory INFO] [Validation] {match_recall 9.627E-01, match_precision 8.250E-01, accuracy 9.648E-01, average_precision 7.975E-01, loss/total 1.388E-01, loss/last 1.388E-01, loss/assignment_nll 1.388E-01, loss/nll_pos 1.916E-01, loss/nll_neg 8.607E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.694E-01}
[06/14/2024 18:56:09 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 18:56:10 gluefactory INFO] New best val: loss/total=0.13884543532703453
[06/14/2024 19:03:38 gluefactory INFO] [Validation] {match_recall 9.619E-01, match_precision 8.123E-01, accuracy 9.608E-01, average_precision 7.845E-01, loss/total 1.477E-01, loss/last 1.477E-01, loss/assignment_nll 1.477E-01, loss/nll_pos 1.996E-01, loss/nll_neg 9.581E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.638E-01}
[06/14/2024 19:03:40 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_20_32444.tar
[06/14/2024 19:03:40 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_12_20084.tar
[06/14/2024 19:03:40 gluefactory INFO] Starting epoch 21
[06/14/2024 19:03:40 gluefactory INFO] lr changed from 7.943282347242815e-05 to 6.309573444801932e-05
[06/14/2024 19:03:46 gluefactory INFO] [E 21 | it 0] loss {total 6.932E-01, last 2.357E-01, assignment_nll 2.357E-01, nll_pos 3.605E-01, nll_neg 1.109E-01, num_matchable 1.260E+02, num_unmatchable 3.713E+02, confidence 1.766E-01, row_norm 9.602E-01}
[06/14/2024 19:07:09 gluefactory INFO] [E 21 | it 100] loss {total 5.646E-01, last 1.346E-01, assignment_nll 1.346E-01, nll_pos 1.986E-01, nll_neg 7.069E-02, num_matchable 1.209E+02, num_unmatchable 3.768E+02, confidence 1.750E-01, row_norm 9.757E-01}
[06/14/2024 19:10:31 gluefactory INFO] [E 21 | it 200] loss {total 5.249E-01, last 8.269E-02, assignment_nll 8.269E-02, nll_pos 8.873E-02, nll_neg 7.664E-02, num_matchable 1.334E+02, num_unmatchable 3.588E+02, confidence 1.812E-01, row_norm 9.769E-01}
[06/14/2024 19:13:54 gluefactory INFO] [E 21 | it 300] loss {total 5.456E-01, last 9.862E-02, assignment_nll 9.862E-02, nll_pos 1.091E-01, nll_neg 8.813E-02, num_matchable 1.240E+02, num_unmatchable 3.744E+02, confidence 1.739E-01, row_norm 9.771E-01}
[06/14/2024 19:17:17 gluefactory INFO] [E 21 | it 400] loss {total 6.127E-01, last 1.860E-01, assignment_nll 1.860E-01, nll_pos 2.725E-01, nll_neg 9.949E-02, num_matchable 1.168E+02, num_unmatchable 3.816E+02, confidence 1.562E-01, row_norm 9.646E-01}
[06/14/2024 19:20:40 gluefactory INFO] [E 21 | it 500] loss {total 5.547E-01, last 1.275E-01, assignment_nll 1.275E-01, nll_pos 1.695E-01, nll_neg 8.548E-02, num_matchable 1.305E+02, num_unmatchable 3.678E+02, confidence 1.723E-01, row_norm 9.705E-01}
[06/14/2024 19:26:34 gluefactory INFO] [Validation] {match_recall 9.650E-01, match_precision 8.334E-01, accuracy 9.673E-01, average_precision 8.065E-01, loss/total 1.288E-01, loss/last 1.288E-01, loss/assignment_nll 1.288E-01, loss/nll_pos 1.765E-01, loss/nll_neg 8.107E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.725E-01}
[06/14/2024 19:26:35 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 19:26:35 gluefactory INFO] New best val: loss/total=0.12879735990740437
[06/14/2024 19:29:58 gluefactory INFO] [E 21 | it 600] loss {total 4.871E-01, last 8.293E-02, assignment_nll 8.293E-02, nll_pos 9.359E-02, nll_neg 7.226E-02, num_matchable 1.315E+02, num_unmatchable 3.643E+02, confidence 1.694E-01, row_norm 9.817E-01}
[06/14/2024 19:33:20 gluefactory INFO] [E 21 | it 700] loss {total 5.662E-01, last 1.292E-01, assignment_nll 1.292E-01, nll_pos 1.775E-01, nll_neg 8.081E-02, num_matchable 1.322E+02, num_unmatchable 3.666E+02, confidence 1.665E-01, row_norm 9.765E-01}
[06/14/2024 19:36:42 gluefactory INFO] [E 21 | it 800] loss {total 5.041E-01, last 9.960E-02, assignment_nll 9.960E-02, nll_pos 1.215E-01, nll_neg 7.774E-02, num_matchable 1.268E+02, num_unmatchable 3.711E+02, confidence 1.703E-01, row_norm 9.752E-01}
[06/14/2024 19:40:05 gluefactory INFO] [E 21 | it 900] loss {total 5.383E-01, last 8.957E-02, assignment_nll 8.957E-02, nll_pos 8.675E-02, nll_neg 9.240E-02, num_matchable 1.274E+02, num_unmatchable 3.712E+02, confidence 1.790E-01, row_norm 9.746E-01}
[06/14/2024 19:43:28 gluefactory INFO] [E 21 | it 1000] loss {total 4.754E-01, last 7.593E-02, assignment_nll 7.593E-02, nll_pos 8.743E-02, nll_neg 6.442E-02, num_matchable 1.462E+02, num_unmatchable 3.520E+02, confidence 1.707E-01, row_norm 9.795E-01}
[06/14/2024 19:49:24 gluefactory INFO] [Validation] {match_recall 9.643E-01, match_precision 8.339E-01, accuracy 9.674E-01, average_precision 8.068E-01, loss/total 1.314E-01, loss/last 1.314E-01, loss/assignment_nll 1.314E-01, loss/nll_pos 1.828E-01, loss/nll_neg 8.010E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.730E-01}
[06/14/2024 19:52:48 gluefactory INFO] [E 21 | it 1100] loss {total 5.727E-01, last 1.698E-01, assignment_nll 1.698E-01, nll_pos 2.441E-01, nll_neg 9.555E-02, num_matchable 1.382E+02, num_unmatchable 3.588E+02, confidence 1.766E-01, row_norm 9.722E-01}
[06/14/2024 19:56:11 gluefactory INFO] [E 21 | it 1200] loss {total 5.429E-01, last 1.224E-01, assignment_nll 1.224E-01, nll_pos 1.750E-01, nll_neg 6.978E-02, num_matchable 1.315E+02, num_unmatchable 3.634E+02, confidence 1.763E-01, row_norm 9.704E-01}
[06/14/2024 19:59:34 gluefactory INFO] [E 21 | it 1300] loss {total 5.611E-01, last 1.152E-01, assignment_nll 1.152E-01, nll_pos 1.411E-01, nll_neg 8.931E-02, num_matchable 1.396E+02, num_unmatchable 3.559E+02, confidence 1.828E-01, row_norm 9.672E-01}
[06/14/2024 20:02:57 gluefactory INFO] [E 21 | it 1400] loss {total 5.268E-01, last 9.764E-02, assignment_nll 9.764E-02, nll_pos 1.162E-01, nll_neg 7.911E-02, num_matchable 1.347E+02, num_unmatchable 3.651E+02, confidence 1.673E-01, row_norm 9.830E-01}
[06/14/2024 20:06:20 gluefactory INFO] [E 21 | it 1500] loss {total 4.920E-01, last 8.861E-02, assignment_nll 8.861E-02, nll_pos 1.060E-01, nll_neg 7.118E-02, num_matchable 1.334E+02, num_unmatchable 3.629E+02, confidence 1.656E-01, row_norm 9.732E-01}
[06/14/2024 20:12:16 gluefactory INFO] [Validation] {match_recall 9.650E-01, match_precision 8.264E-01, accuracy 9.653E-01, average_precision 8.002E-01, loss/total 1.313E-01, loss/last 1.313E-01, loss/assignment_nll 1.313E-01, loss/nll_pos 1.760E-01, loss/nll_neg 8.651E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.706E-01}
[06/14/2024 20:19:43 gluefactory INFO] [Validation] {match_recall 9.653E-01, match_precision 8.182E-01, accuracy 9.628E-01, average_precision 7.923E-01, loss/total 1.341E-01, loss/last 1.341E-01, loss/assignment_nll 1.341E-01, loss/nll_pos 1.712E-01, loss/nll_neg 9.698E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.720E-01}
[06/14/2024 20:19:45 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_21_33989.tar
[06/14/2024 20:19:46 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_13_21629.tar
[06/14/2024 20:19:46 gluefactory INFO] Starting epoch 22
[06/14/2024 20:19:46 gluefactory INFO] lr changed from 6.309573444801932e-05 to 5.0118723362727224e-05
[06/14/2024 20:19:51 gluefactory INFO] [E 22 | it 0] loss {total 5.894E-01, last 1.164E-01, assignment_nll 1.164E-01, nll_pos 1.316E-01, nll_neg 1.012E-01, num_matchable 1.252E+02, num_unmatchable 3.714E+02, confidence 1.795E-01, row_norm 9.711E-01}
[06/14/2024 20:23:14 gluefactory INFO] [E 22 | it 100] loss {total 6.240E-01, last 2.355E-01, assignment_nll 2.355E-01, nll_pos 3.691E-01, nll_neg 1.019E-01, num_matchable 1.209E+02, num_unmatchable 3.785E+02, confidence 1.619E-01, row_norm 9.592E-01}
[06/14/2024 20:26:37 gluefactory INFO] [E 22 | it 200] loss {total 4.777E-01, last 7.492E-02, assignment_nll 7.492E-02, nll_pos 8.046E-02, nll_neg 6.938E-02, num_matchable 1.231E+02, num_unmatchable 3.719E+02, confidence 1.618E-01, row_norm 9.814E-01}
[06/14/2024 20:29:59 gluefactory INFO] [E 22 | it 300] loss {total 4.558E-01, last 6.177E-02, assignment_nll 6.177E-02, nll_pos 6.590E-02, nll_neg 5.765E-02, num_matchable 1.160E+02, num_unmatchable 3.838E+02, confidence 1.550E-01, row_norm 9.839E-01}
[06/14/2024 20:33:21 gluefactory INFO] [E 22 | it 400] loss {total 6.530E-01, last 1.734E-01, assignment_nll 1.734E-01, nll_pos 2.576E-01, nll_neg 8.926E-02, num_matchable 1.177E+02, num_unmatchable 3.777E+02, confidence 1.760E-01, row_norm 9.677E-01}
[06/14/2024 20:36:43 gluefactory INFO] [E 22 | it 500] loss {total 5.067E-01, last 8.969E-02, assignment_nll 8.969E-02, nll_pos 1.019E-01, nll_neg 7.753E-02, num_matchable 1.329E+02, num_unmatchable 3.656E+02, confidence 1.684E-01, row_norm 9.745E-01}
[06/14/2024 20:42:37 gluefactory INFO] [Validation] {match_recall 9.667E-01, match_precision 8.271E-01, accuracy 9.655E-01, average_precision 8.017E-01, loss/total 1.250E-01, loss/last 1.250E-01, loss/assignment_nll 1.250E-01, loss/nll_pos 1.645E-01, loss/nll_neg 8.548E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.723E-01}
[06/14/2024 20:42:37 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 20:42:37 gluefactory INFO] New best val: loss/total=0.1249891611920932
[06/14/2024 20:46:01 gluefactory INFO] [E 22 | it 600] loss {total 4.548E-01, last 6.685E-02, assignment_nll 6.685E-02, nll_pos 7.175E-02, nll_neg 6.195E-02, num_matchable 1.278E+02, num_unmatchable 3.678E+02, confidence 1.630E-01, row_norm 9.834E-01}
[06/14/2024 20:49:24 gluefactory INFO] [E 22 | it 700] loss {total 4.659E-01, last 6.941E-02, assignment_nll 6.941E-02, nll_pos 7.522E-02, nll_neg 6.361E-02, num_matchable 1.319E+02, num_unmatchable 3.669E+02, confidence 1.587E-01, row_norm 9.821E-01}
[06/14/2024 20:52:47 gluefactory INFO] [E 22 | it 800] loss {total 5.369E-01, last 1.040E-01, assignment_nll 1.040E-01, nll_pos 1.176E-01, nll_neg 9.041E-02, num_matchable 1.249E+02, num_unmatchable 3.724E+02, confidence 1.734E-01, row_norm 9.740E-01}
[06/14/2024 20:56:10 gluefactory INFO] [E 22 | it 900] loss {total 6.270E-01, last 1.571E-01, assignment_nll 1.571E-01, nll_pos 2.164E-01, nll_neg 9.793E-02, num_matchable 1.237E+02, num_unmatchable 3.751E+02, confidence 1.761E-01, row_norm 9.665E-01}
[06/14/2024 20:59:33 gluefactory INFO] [E 22 | it 1000] loss {total 4.946E-01, last 9.910E-02, assignment_nll 9.910E-02, nll_pos 1.231E-01, nll_neg 7.507E-02, num_matchable 1.358E+02, num_unmatchable 3.618E+02, confidence 1.609E-01, row_norm 9.737E-01}
[06/14/2024 21:05:29 gluefactory INFO] [Validation] {match_recall 9.657E-01, match_precision 8.359E-01, accuracy 9.681E-01, average_precision 8.099E-01, loss/total 1.251E-01, loss/last 1.251E-01, loss/assignment_nll 1.251E-01, loss/nll_pos 1.701E-01, loss/nll_neg 8.004E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.725E-01}
[06/14/2024 21:05:50 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_22_35000.tar
[06/14/2024 21:05:51 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_14_23174.tar
[06/14/2024 21:08:54 gluefactory INFO] [E 22 | it 1100] loss {total 4.914E-01, last 1.018E-01, assignment_nll 1.018E-01, nll_pos 1.251E-01, nll_neg 7.859E-02, num_matchable 1.369E+02, num_unmatchable 3.620E+02, confidence 1.650E-01, row_norm 9.802E-01}
[06/14/2024 21:12:17 gluefactory INFO] [E 22 | it 1200] loss {total 4.450E-01, last 5.895E-02, assignment_nll 5.895E-02, nll_pos 5.841E-02, nll_neg 5.949E-02, num_matchable 1.265E+02, num_unmatchable 3.711E+02, confidence 1.570E-01, row_norm 9.845E-01}
[06/14/2024 21:15:39 gluefactory INFO] [E 22 | it 1300] loss {total 6.444E-01, last 1.242E-01, assignment_nll 1.242E-01, nll_pos 1.579E-01, nll_neg 9.057E-02, num_matchable 1.217E+02, num_unmatchable 3.720E+02, confidence 1.888E-01, row_norm 9.651E-01}
[06/14/2024 21:19:02 gluefactory INFO] [E 22 | it 1400] loss {total 4.755E-01, last 7.513E-02, assignment_nll 7.513E-02, nll_pos 8.162E-02, nll_neg 6.863E-02, num_matchable 1.401E+02, num_unmatchable 3.583E+02, confidence 1.677E-01, row_norm 9.806E-01}
[06/14/2024 21:22:25 gluefactory INFO] [E 22 | it 1500] loss {total 5.429E-01, last 1.377E-01, assignment_nll 1.377E-01, nll_pos 1.898E-01, nll_neg 8.558E-02, num_matchable 1.511E+02, num_unmatchable 3.391E+02, confidence 1.770E-01, row_norm 9.581E-01}
[06/14/2024 21:28:20 gluefactory INFO] [Validation] {match_recall 9.617E-01, match_precision 8.226E-01, accuracy 9.641E-01, average_precision 7.954E-01, loss/total 1.450E-01, loss/last 1.450E-01, loss/assignment_nll 1.450E-01, loss/nll_pos 1.966E-01, loss/nll_neg 9.331E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.650E-01}
[06/14/2024 21:35:48 gluefactory INFO] [Validation] {match_recall 9.668E-01, match_precision 8.429E-01, accuracy 9.698E-01, average_precision 8.161E-01, loss/total 1.209E-01, loss/last 1.209E-01, loss/assignment_nll 1.209E-01, loss/nll_pos 1.697E-01, loss/nll_neg 7.209E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.747E-01}
[06/14/2024 21:35:48 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 21:35:49 gluefactory INFO] New best val: loss/total=0.12088618107446814
[06/14/2024 21:35:51 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_22_35534.tar
[06/14/2024 21:35:51 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_15_24719.tar
[06/14/2024 21:35:51 gluefactory INFO] Starting epoch 23
[06/14/2024 21:35:51 gluefactory INFO] lr changed from 5.0118723362727224e-05 to 3.981071705534972e-05
[06/14/2024 21:35:57 gluefactory INFO] [E 23 | it 0] loss {total 4.855E-01, last 8.836E-02, assignment_nll 8.836E-02, nll_pos 1.083E-01, nll_neg 6.841E-02, num_matchable 1.280E+02, num_unmatchable 3.709E+02, confidence 1.659E-01, row_norm 9.811E-01}
[06/14/2024 21:39:19 gluefactory INFO] [E 23 | it 100] loss {total 5.969E-01, last 1.949E-01, assignment_nll 1.949E-01, nll_pos 3.069E-01, nll_neg 8.293E-02, num_matchable 1.285E+02, num_unmatchable 3.691E+02, confidence 1.589E-01, row_norm 9.732E-01}
[06/14/2024 21:42:41 gluefactory INFO] [E 23 | it 200] loss {total 5.489E-01, last 1.347E-01, assignment_nll 1.347E-01, nll_pos 1.920E-01, nll_neg 7.743E-02, num_matchable 1.355E+02, num_unmatchable 3.583E+02, confidence 1.641E-01, row_norm 9.715E-01}
[06/14/2024 21:46:02 gluefactory INFO] [E 23 | it 300] loss {total 5.804E-01, last 1.051E-01, assignment_nll 1.051E-01, nll_pos 1.310E-01, nll_neg 7.915E-02, num_matchable 1.217E+02, num_unmatchable 3.764E+02, confidence 1.594E-01, row_norm 9.725E-01}
[06/14/2024 21:49:24 gluefactory INFO] [E 23 | it 400] loss {total 7.509E-01, last 3.208E-01, assignment_nll 3.208E-01, nll_pos 5.519E-01, nll_neg 8.971E-02, num_matchable 1.219E+02, num_unmatchable 3.748E+02, confidence 1.640E-01, row_norm 9.627E-01}
[06/14/2024 21:52:46 gluefactory INFO] [E 23 | it 500] loss {total 4.979E-01, last 7.482E-02, assignment_nll 7.482E-02, nll_pos 8.365E-02, nll_neg 6.599E-02, num_matchable 1.398E+02, num_unmatchable 3.585E+02, confidence 1.780E-01, row_norm 9.818E-01}
[06/14/2024 21:58:41 gluefactory INFO] [Validation] {match_recall 9.684E-01, match_precision 8.424E-01, accuracy 9.698E-01, average_precision 8.169E-01, loss/total 1.159E-01, loss/last 1.159E-01, loss/assignment_nll 1.159E-01, loss/nll_pos 1.606E-01, loss/nll_neg 7.111E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.756E-01}
[06/14/2024 21:58:41 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 21:58:42 gluefactory INFO] New best val: loss/total=0.11586072103959655
[06/14/2024 22:02:06 gluefactory INFO] [E 23 | it 600] loss {total 4.545E-01, last 6.762E-02, assignment_nll 6.762E-02, nll_pos 7.138E-02, nll_neg 6.387E-02, num_matchable 1.227E+02, num_unmatchable 3.727E+02, confidence 1.632E-01, row_norm 9.832E-01}
[06/14/2024 22:05:29 gluefactory INFO] [E 23 | it 700] loss {total 4.507E-01, last 6.036E-02, assignment_nll 6.036E-02, nll_pos 5.765E-02, nll_neg 6.306E-02, num_matchable 1.306E+02, num_unmatchable 3.677E+02, confidence 1.628E-01, row_norm 9.833E-01}
[06/14/2024 22:08:52 gluefactory INFO] [E 23 | it 800] loss {total 5.435E-01, last 1.328E-01, assignment_nll 1.328E-01, nll_pos 1.778E-01, nll_neg 8.790E-02, num_matchable 1.196E+02, num_unmatchable 3.766E+02, confidence 1.743E-01, row_norm 9.754E-01}
[06/14/2024 22:12:15 gluefactory INFO] [E 23 | it 900] loss {total 5.194E-01, last 1.132E-01, assignment_nll 1.132E-01, nll_pos 1.489E-01, nll_neg 7.748E-02, num_matchable 1.338E+02, num_unmatchable 3.636E+02, confidence 1.673E-01, row_norm 9.723E-01}
[06/14/2024 22:15:37 gluefactory INFO] [E 23 | it 1000] loss {total 5.301E-01, last 1.454E-01, assignment_nll 1.454E-01, nll_pos 2.229E-01, nll_neg 6.785E-02, num_matchable 1.378E+02, num_unmatchable 3.602E+02, confidence 1.560E-01, row_norm 9.719E-01}
[06/14/2024 22:21:32 gluefactory INFO] [Validation] {match_recall 9.683E-01, match_precision 8.461E-01, accuracy 9.709E-01, average_precision 8.206E-01, loss/total 1.150E-01, loss/last 1.150E-01, loss/assignment_nll 1.150E-01, loss/nll_pos 1.578E-01, loss/nll_neg 7.220E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.742E-01}
[06/14/2024 22:21:32 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 22:21:33 gluefactory INFO] New best val: loss/total=0.1150036582975319
[06/14/2024 22:24:57 gluefactory INFO] [E 23 | it 1100] loss {total 4.734E-01, last 7.596E-02, assignment_nll 7.596E-02, nll_pos 8.551E-02, nll_neg 6.641E-02, num_matchable 1.257E+02, num_unmatchable 3.724E+02, confidence 1.644E-01, row_norm 9.776E-01}
[06/14/2024 22:28:20 gluefactory INFO] [E 23 | it 1200] loss {total 5.140E-01, last 1.099E-01, assignment_nll 1.099E-01, nll_pos 1.477E-01, nll_neg 7.202E-02, num_matchable 1.212E+02, num_unmatchable 3.743E+02, confidence 1.647E-01, row_norm 9.790E-01}
[06/14/2024 22:31:43 gluefactory INFO] [E 23 | it 1300] loss {total 5.158E-01, last 8.931E-02, assignment_nll 8.931E-02, nll_pos 1.119E-01, nll_neg 6.676E-02, num_matchable 1.260E+02, num_unmatchable 3.699E+02, confidence 1.654E-01, row_norm 9.783E-01}
[06/14/2024 22:35:06 gluefactory INFO] [E 23 | it 1400] loss {total 4.966E-01, last 8.804E-02, assignment_nll 8.804E-02, nll_pos 1.149E-01, nll_neg 6.121E-02, num_matchable 1.464E+02, num_unmatchable 3.513E+02, confidence 1.622E-01, row_norm 9.753E-01}
[06/14/2024 22:38:28 gluefactory INFO] [E 23 | it 1500] loss {total 4.899E-01, last 9.002E-02, assignment_nll 9.002E-02, nll_pos 1.175E-01, nll_neg 6.257E-02, num_matchable 1.458E+02, num_unmatchable 3.484E+02, confidence 1.726E-01, row_norm 9.771E-01}
[06/14/2024 22:44:26 gluefactory INFO] [Validation] {match_recall 9.679E-01, match_precision 8.449E-01, accuracy 9.706E-01, average_precision 8.194E-01, loss/total 1.155E-01, loss/last 1.155E-01, loss/assignment_nll 1.155E-01, loss/nll_pos 1.560E-01, loss/nll_neg 7.508E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.749E-01}
[06/14/2024 22:51:53 gluefactory INFO] [Validation] {match_recall 9.680E-01, match_precision 8.373E-01, accuracy 9.685E-01, average_precision 8.120E-01, loss/total 1.166E-01, loss/last 1.166E-01, loss/assignment_nll 1.166E-01, loss/nll_pos 1.595E-01, loss/nll_neg 7.362E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.727E-01}
[06/14/2024 22:51:55 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_23_37079.tar
[06/14/2024 22:51:55 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_16_25000.tar
[06/14/2024 22:51:55 gluefactory INFO] Starting epoch 24
[06/14/2024 22:51:55 gluefactory INFO] lr changed from 3.981071705534972e-05 to 3.162277660168379e-05
[06/14/2024 22:52:01 gluefactory INFO] [E 24 | it 0] loss {total 6.226E-01, last 1.998E-01, assignment_nll 1.998E-01, nll_pos 3.081E-01, nll_neg 9.157E-02, num_matchable 1.230E+02, num_unmatchable 3.755E+02, confidence 1.617E-01, row_norm 9.644E-01}
[06/14/2024 22:55:22 gluefactory INFO] [E 24 | it 100] loss {total 5.166E-01, last 9.539E-02, assignment_nll 9.539E-02, nll_pos 1.198E-01, nll_neg 7.095E-02, num_matchable 1.424E+02, num_unmatchable 3.536E+02, confidence 1.716E-01, row_norm 9.766E-01}
[06/14/2024 22:58:44 gluefactory INFO] [E 24 | it 200] loss {total 4.548E-01, last 5.695E-02, assignment_nll 5.695E-02, nll_pos 5.769E-02, nll_neg 5.620E-02, num_matchable 1.330E+02, num_unmatchable 3.616E+02, confidence 1.621E-01, row_norm 9.788E-01}
[06/14/2024 23:02:06 gluefactory INFO] [E 24 | it 300] loss {total 4.863E-01, last 6.368E-02, assignment_nll 6.368E-02, nll_pos 5.999E-02, nll_neg 6.736E-02, num_matchable 1.274E+02, num_unmatchable 3.704E+02, confidence 1.694E-01, row_norm 9.800E-01}
[06/14/2024 23:05:29 gluefactory INFO] [E 24 | it 400] loss {total 4.702E-01, last 7.155E-02, assignment_nll 7.155E-02, nll_pos 7.708E-02, nll_neg 6.602E-02, num_matchable 1.140E+02, num_unmatchable 3.832E+02, confidence 1.598E-01, row_norm 9.815E-01}
[06/14/2024 23:08:52 gluefactory INFO] [E 24 | it 500] loss {total 6.386E-01, last 2.331E-01, assignment_nll 2.331E-01, nll_pos 3.798E-01, nll_neg 8.645E-02, num_matchable 1.325E+02, num_unmatchable 3.662E+02, confidence 1.630E-01, row_norm 9.670E-01}
[06/14/2024 23:14:48 gluefactory INFO] [Validation] {match_recall 9.695E-01, match_precision 8.407E-01, accuracy 9.695E-01, average_precision 8.162E-01, loss/total 1.117E-01, loss/last 1.117E-01, loss/assignment_nll 1.117E-01, loss/nll_pos 1.476E-01, loss/nll_neg 7.587E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.746E-01}
[06/14/2024 23:14:48 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 23:14:48 gluefactory INFO] New best val: loss/total=0.11174533047607046
[06/14/2024 23:18:12 gluefactory INFO] [E 24 | it 600] loss {total 4.877E-01, last 9.095E-02, assignment_nll 9.095E-02, nll_pos 1.198E-01, nll_neg 6.211E-02, num_matchable 1.196E+02, num_unmatchable 3.753E+02, confidence 1.644E-01, row_norm 9.811E-01}
[06/14/2024 23:21:35 gluefactory INFO] [E 24 | it 700] loss {total 5.496E-01, last 1.504E-01, assignment_nll 1.504E-01, nll_pos 2.237E-01, nll_neg 7.696E-02, num_matchable 1.294E+02, num_unmatchable 3.702E+02, confidence 1.567E-01, row_norm 9.774E-01}
[06/14/2024 23:24:58 gluefactory INFO] [E 24 | it 800] loss {total 5.198E-01, last 1.085E-01, assignment_nll 1.085E-01, nll_pos 1.408E-01, nll_neg 7.628E-02, num_matchable 1.144E+02, num_unmatchable 3.836E+02, confidence 1.632E-01, row_norm 9.779E-01}
[06/14/2024 23:28:21 gluefactory INFO] [E 24 | it 900] loss {total 7.216E-01, last 3.187E-01, assignment_nll 3.187E-01, nll_pos 5.510E-01, nll_neg 8.646E-02, num_matchable 1.278E+02, num_unmatchable 3.688E+02, confidence 1.634E-01, row_norm 9.562E-01}
[06/14/2024 23:31:44 gluefactory INFO] [E 24 | it 1000] loss {total 4.315E-01, last 6.454E-02, assignment_nll 6.454E-02, nll_pos 7.920E-02, nll_neg 4.987E-02, num_matchable 1.416E+02, num_unmatchable 3.557E+02, confidence 1.575E-01, row_norm 9.808E-01}
[06/14/2024 23:37:40 gluefactory INFO] [Validation] {match_recall 9.693E-01, match_precision 8.486E-01, accuracy 9.716E-01, average_precision 8.235E-01, loss/total 1.099E-01, loss/last 1.099E-01, loss/assignment_nll 1.099E-01, loss/nll_pos 1.507E-01, loss/nll_neg 6.923E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.769E-01}
[06/14/2024 23:37:40 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/14/2024 23:37:41 gluefactory INFO] New best val: loss/total=0.10994246292777657
[06/14/2024 23:41:05 gluefactory INFO] [E 24 | it 1100] loss {total 4.175E-01, last 5.218E-02, assignment_nll 5.218E-02, nll_pos 5.279E-02, nll_neg 5.157E-02, num_matchable 1.357E+02, num_unmatchable 3.635E+02, confidence 1.610E-01, row_norm 9.859E-01}
[06/14/2024 23:44:28 gluefactory INFO] [E 24 | it 1200] loss {total 5.020E-01, last 8.403E-02, assignment_nll 8.403E-02, nll_pos 9.193E-02, nll_neg 7.613E-02, num_matchable 1.180E+02, num_unmatchable 3.776E+02, confidence 1.659E-01, row_norm 9.772E-01}
[06/14/2024 23:47:51 gluefactory INFO] [E 24 | it 1300] loss {total 5.342E-01, last 1.085E-01, assignment_nll 1.085E-01, nll_pos 1.468E-01, nll_neg 7.010E-02, num_matchable 1.295E+02, num_unmatchable 3.669E+02, confidence 1.628E-01, row_norm 9.762E-01}
[06/14/2024 23:51:14 gluefactory INFO] [E 24 | it 1400] loss {total 6.387E-01, last 2.239E-01, assignment_nll 2.239E-01, nll_pos 3.645E-01, nll_neg 8.330E-02, num_matchable 1.396E+02, num_unmatchable 3.582E+02, confidence 1.527E-01, row_norm 9.628E-01}
[06/14/2024 23:54:37 gluefactory INFO] [E 24 | it 1500] loss {total 4.636E-01, last 7.317E-02, assignment_nll 7.317E-02, nll_pos 7.979E-02, nll_neg 6.655E-02, num_matchable 1.412E+02, num_unmatchable 3.529E+02, confidence 1.711E-01, row_norm 9.804E-01}
[06/15/2024 00:00:31 gluefactory INFO] [Validation] {match_recall 9.690E-01, match_precision 8.460E-01, accuracy 9.709E-01, average_precision 8.208E-01, loss/total 1.111E-01, loss/last 1.111E-01, loss/assignment_nll 1.111E-01, loss/nll_pos 1.498E-01, loss/nll_neg 7.233E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.750E-01}
[06/15/2024 00:07:57 gluefactory INFO] [Validation] {match_recall 9.679E-01, match_precision 8.455E-01, accuracy 9.709E-01, average_precision 8.199E-01, loss/total 1.158E-01, loss/last 1.158E-01, loss/assignment_nll 1.158E-01, loss/nll_pos 1.654E-01, loss/nll_neg 6.626E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.718E-01}
[06/15/2024 00:07:59 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_24_38624.tar
[06/15/2024 00:07:59 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_16_26264.tar
[06/15/2024 00:07:59 gluefactory INFO] Starting epoch 25
[06/15/2024 00:07:59 gluefactory INFO] lr changed from 3.162277660168379e-05 to 2.5118864315095798e-05
[06/15/2024 00:08:05 gluefactory INFO] [E 25 | it 0] loss {total 4.963E-01, last 9.367E-02, assignment_nll 9.367E-02, nll_pos 1.220E-01, nll_neg 6.535E-02, num_matchable 1.158E+02, num_unmatchable 3.824E+02, confidence 1.628E-01, row_norm 9.795E-01}
[06/15/2024 00:11:27 gluefactory INFO] [E 25 | it 100] loss {total 4.720E-01, last 8.821E-02, assignment_nll 8.821E-02, nll_pos 1.118E-01, nll_neg 6.458E-02, num_matchable 1.241E+02, num_unmatchable 3.756E+02, confidence 1.522E-01, row_norm 9.785E-01}
[06/15/2024 00:14:50 gluefactory INFO] [E 25 | it 200] loss {total 4.406E-01, last 6.912E-02, assignment_nll 6.912E-02, nll_pos 8.507E-02, nll_neg 5.317E-02, num_matchable 1.232E+02, num_unmatchable 3.742E+02, confidence 1.502E-01, row_norm 9.822E-01}
[06/15/2024 00:18:13 gluefactory INFO] [E 25 | it 300] loss {total 5.216E-01, last 1.538E-01, assignment_nll 1.538E-01, nll_pos 2.359E-01, nll_neg 7.168E-02, num_matchable 1.278E+02, num_unmatchable 3.702E+02, confidence 1.556E-01, row_norm 9.737E-01}
[06/15/2024 00:21:35 gluefactory INFO] [E 25 | it 400] loss {total 6.597E-01, last 1.672E-01, assignment_nll 1.672E-01, nll_pos 2.552E-01, nll_neg 7.911E-02, num_matchable 1.237E+02, num_unmatchable 3.732E+02, confidence 1.590E-01, row_norm 9.607E-01}
[06/15/2024 00:24:58 gluefactory INFO] [E 25 | it 500] loss {total 4.294E-01, last 5.700E-02, assignment_nll 5.700E-02, nll_pos 5.730E-02, nll_neg 5.669E-02, num_matchable 1.341E+02, num_unmatchable 3.650E+02, confidence 1.595E-01, row_norm 9.852E-01}
[06/15/2024 00:30:54 gluefactory INFO] [Validation] {match_recall 9.708E-01, match_precision 8.513E-01, accuracy 9.723E-01, average_precision 8.268E-01, loss/total 1.045E-01, loss/last 1.045E-01, loss/assignment_nll 1.045E-01, loss/nll_pos 1.415E-01, loss/nll_neg 6.738E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.764E-01}
[06/15/2024 00:30:54 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 00:30:55 gluefactory INFO] New best val: loss/total=0.10445622220294314
[06/15/2024 00:34:19 gluefactory INFO] [E 25 | it 600] loss {total 4.766E-01, last 7.546E-02, assignment_nll 7.546E-02, nll_pos 9.020E-02, nll_neg 6.073E-02, num_matchable 1.166E+02, num_unmatchable 3.772E+02, confidence 1.635E-01, row_norm 9.796E-01}
[06/15/2024 00:37:42 gluefactory INFO] [E 25 | it 700] loss {total 4.668E-01, last 7.026E-02, assignment_nll 7.026E-02, nll_pos 8.740E-02, nll_neg 5.312E-02, num_matchable 1.290E+02, num_unmatchable 3.698E+02, confidence 1.543E-01, row_norm 9.832E-01}
[06/15/2024 00:41:05 gluefactory INFO] [E 25 | it 800] loss {total 4.915E-01, last 6.353E-02, assignment_nll 6.353E-02, nll_pos 6.733E-02, nll_neg 5.973E-02, num_matchable 1.152E+02, num_unmatchable 3.819E+02, confidence 1.665E-01, row_norm 9.840E-01}
[06/15/2024 00:44:27 gluefactory INFO] [E 25 | it 900] loss {total 4.780E-01, last 6.665E-02, assignment_nll 6.665E-02, nll_pos 7.231E-02, nll_neg 6.100E-02, num_matchable 1.384E+02, num_unmatchable 3.582E+02, confidence 1.668E-01, row_norm 9.781E-01}
[06/15/2024 00:47:50 gluefactory INFO] [E 25 | it 1000] loss {total 4.243E-01, last 7.105E-02, assignment_nll 7.105E-02, nll_pos 7.907E-02, nll_neg 6.304E-02, num_matchable 1.362E+02, num_unmatchable 3.604E+02, confidence 1.571E-01, row_norm 9.728E-01}
[06/15/2024 00:53:46 gluefactory INFO] [Validation] {match_recall 9.708E-01, match_precision 8.497E-01, accuracy 9.720E-01, average_precision 8.255E-01, loss/total 1.046E-01, loss/last 1.046E-01, loss/assignment_nll 1.046E-01, loss/nll_pos 1.420E-01, loss/nll_neg 6.716E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.766E-01}
[06/15/2024 00:57:10 gluefactory INFO] [E 25 | it 1100] loss {total 4.257E-01, last 6.994E-02, assignment_nll 6.994E-02, nll_pos 7.574E-02, nll_neg 6.414E-02, num_matchable 1.327E+02, num_unmatchable 3.668E+02, confidence 1.556E-01, row_norm 9.822E-01}
[06/15/2024 01:00:33 gluefactory INFO] [E 25 | it 1200] loss {total 5.037E-01, last 9.252E-02, assignment_nll 9.252E-02, nll_pos 1.175E-01, nll_neg 6.753E-02, num_matchable 1.279E+02, num_unmatchable 3.678E+02, confidence 1.646E-01, row_norm 9.767E-01}
[06/15/2024 01:03:56 gluefactory INFO] [E 25 | it 1300] loss {total 4.830E-01, last 6.673E-02, assignment_nll 6.673E-02, nll_pos 6.866E-02, nll_neg 6.481E-02, num_matchable 1.304E+02, num_unmatchable 3.662E+02, confidence 1.683E-01, row_norm 9.793E-01}
[06/15/2024 01:06:28 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_25_40000.tar
[06/15/2024 01:06:28 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_17_27809.tar
[06/15/2024 01:07:19 gluefactory INFO] [E 25 | it 1400] loss {total 6.052E-01, last 1.918E-01, assignment_nll 1.918E-01, nll_pos 3.085E-01, nll_neg 7.504E-02, num_matchable 1.436E+02, num_unmatchable 3.544E+02, confidence 1.566E-01, row_norm 9.726E-01}
[06/15/2024 01:10:42 gluefactory INFO] [E 25 | it 1500] loss {total 4.510E-01, last 6.080E-02, assignment_nll 6.080E-02, nll_pos 6.834E-02, nll_neg 5.325E-02, num_matchable 1.452E+02, num_unmatchable 3.504E+02, confidence 1.654E-01, row_norm 9.769E-01}
[06/15/2024 01:16:37 gluefactory INFO] [Validation] {match_recall 9.707E-01, match_precision 8.553E-01, accuracy 9.734E-01, average_precision 8.306E-01, loss/total 1.032E-01, loss/last 1.032E-01, loss/assignment_nll 1.032E-01, loss/nll_pos 1.439E-01, loss/nll_neg 6.239E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.769E-01}
[06/15/2024 01:16:37 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 01:16:37 gluefactory INFO] New best val: loss/total=0.10315867603514599
[06/15/2024 01:24:03 gluefactory INFO] [Validation] {match_recall 9.707E-01, match_precision 8.498E-01, accuracy 9.720E-01, average_precision 8.252E-01, loss/total 1.052E-01, loss/last 1.052E-01, loss/assignment_nll 1.052E-01, loss/nll_pos 1.435E-01, loss/nll_neg 6.684E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.761E-01}
[06/15/2024 01:24:05 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_25_40169.tar
[06/15/2024 01:24:05 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_18_29354.tar
[06/15/2024 01:24:05 gluefactory INFO] Starting epoch 26
[06/15/2024 01:24:05 gluefactory INFO] lr changed from 2.5118864315095798e-05 to 1.9952623149688793e-05
[06/15/2024 01:24:11 gluefactory INFO] [E 26 | it 0] loss {total 4.747E-01, last 7.080E-02, assignment_nll 7.080E-02, nll_pos 7.400E-02, nll_neg 6.761E-02, num_matchable 1.304E+02, num_unmatchable 3.670E+02, confidence 1.698E-01, row_norm 9.817E-01}
[06/15/2024 01:27:33 gluefactory INFO] [E 26 | it 100] loss {total 4.598E-01, last 8.058E-02, assignment_nll 8.058E-02, nll_pos 9.856E-02, nll_neg 6.260E-02, num_matchable 1.228E+02, num_unmatchable 3.744E+02, confidence 1.542E-01, row_norm 9.794E-01}
[06/15/2024 01:30:56 gluefactory INFO] [E 26 | it 200] loss {total 4.464E-01, last 5.511E-02, assignment_nll 5.511E-02, nll_pos 5.713E-02, nll_neg 5.309E-02, num_matchable 1.323E+02, num_unmatchable 3.618E+02, confidence 1.623E-01, row_norm 9.832E-01}
[06/15/2024 01:34:19 gluefactory INFO] [E 26 | it 300] loss {total 5.023E-01, last 8.697E-02, assignment_nll 8.697E-02, nll_pos 1.013E-01, nll_neg 7.260E-02, num_matchable 1.256E+02, num_unmatchable 3.728E+02, confidence 1.597E-01, row_norm 9.778E-01}
[06/15/2024 01:37:41 gluefactory INFO] [E 26 | it 400] loss {total 4.617E-01, last 6.133E-02, assignment_nll 6.133E-02, nll_pos 6.159E-02, nll_neg 6.108E-02, num_matchable 1.200E+02, num_unmatchable 3.783E+02, confidence 1.553E-01, row_norm 9.835E-01}
[06/15/2024 01:41:04 gluefactory INFO] [E 26 | it 500] loss {total 4.647E-01, last 7.862E-02, assignment_nll 7.862E-02, nll_pos 1.024E-01, nll_neg 5.485E-02, num_matchable 1.371E+02, num_unmatchable 3.610E+02, confidence 1.634E-01, row_norm 9.806E-01}
[06/15/2024 01:47:00 gluefactory INFO] [Validation] {match_recall 9.711E-01, match_precision 8.597E-01, accuracy 9.745E-01, average_precision 8.351E-01, loss/total 1.021E-01, loss/last 1.021E-01, loss/assignment_nll 1.021E-01, loss/nll_pos 1.456E-01, loss/nll_neg 5.856E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.780E-01}
[06/15/2024 01:47:00 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 01:47:01 gluefactory INFO] New best val: loss/total=0.10207871655801502
[06/15/2024 01:50:25 gluefactory INFO] [E 26 | it 600] loss {total 5.120E-01, last 1.069E-01, assignment_nll 1.069E-01, nll_pos 1.509E-01, nll_neg 6.292E-02, num_matchable 1.180E+02, num_unmatchable 3.769E+02, confidence 1.619E-01, row_norm 9.800E-01}
[06/15/2024 01:53:48 gluefactory INFO] [E 26 | it 700] loss {total 5.122E-01, last 1.138E-01, assignment_nll 1.138E-01, nll_pos 1.586E-01, nll_neg 6.903E-02, num_matchable 1.156E+02, num_unmatchable 3.836E+02, confidence 1.518E-01, row_norm 9.753E-01}
[06/15/2024 01:57:11 gluefactory INFO] [E 26 | it 800] loss {total 5.270E-01, last 1.116E-01, assignment_nll 1.116E-01, nll_pos 1.481E-01, nll_neg 7.506E-02, num_matchable 1.267E+02, num_unmatchable 3.699E+02, confidence 1.624E-01, row_norm 9.768E-01}
[06/15/2024 02:00:34 gluefactory INFO] [E 26 | it 900] loss {total 5.894E-01, last 1.716E-01, assignment_nll 1.716E-01, nll_pos 2.692E-01, nll_neg 7.406E-02, num_matchable 1.210E+02, num_unmatchable 3.780E+02, confidence 1.582E-01, row_norm 9.706E-01}
[06/15/2024 02:03:57 gluefactory INFO] [E 26 | it 1000] loss {total 5.566E-01, last 1.408E-01, assignment_nll 1.408E-01, nll_pos 1.982E-01, nll_neg 8.338E-02, num_matchable 1.352E+02, num_unmatchable 3.603E+02, confidence 1.652E-01, row_norm 9.672E-01}
[06/15/2024 02:09:52 gluefactory INFO] [Validation] {match_recall 9.717E-01, match_precision 8.504E-01, accuracy 9.722E-01, average_precision 8.267E-01, loss/total 1.017E-01, loss/last 1.017E-01, loss/assignment_nll 1.017E-01, loss/nll_pos 1.358E-01, loss/nll_neg 6.749E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.767E-01}
[06/15/2024 02:09:52 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 02:09:53 gluefactory INFO] New best val: loss/total=0.1016668164690538
[06/15/2024 02:13:17 gluefactory INFO] [E 26 | it 1100] loss {total 4.287E-01, last 6.247E-02, assignment_nll 6.247E-02, nll_pos 7.090E-02, nll_neg 5.405E-02, num_matchable 1.306E+02, num_unmatchable 3.678E+02, confidence 1.546E-01, row_norm 9.821E-01}
[06/15/2024 02:16:40 gluefactory INFO] [E 26 | it 1200] loss {total 3.902E-01, last 4.529E-02, assignment_nll 4.529E-02, nll_pos 4.771E-02, nll_neg 4.288E-02, num_matchable 1.399E+02, num_unmatchable 3.575E+02, confidence 1.495E-01, row_norm 9.847E-01}
[06/15/2024 02:20:03 gluefactory INFO] [E 26 | it 1300] loss {total 4.927E-01, last 9.612E-02, assignment_nll 9.612E-02, nll_pos 1.326E-01, nll_neg 5.964E-02, num_matchable 1.307E+02, num_unmatchable 3.655E+02, confidence 1.609E-01, row_norm 9.753E-01}
[06/15/2024 02:23:26 gluefactory INFO] [E 26 | it 1400] loss {total 4.219E-01, last 5.679E-02, assignment_nll 5.679E-02, nll_pos 5.959E-02, nll_neg 5.398E-02, num_matchable 1.404E+02, num_unmatchable 3.584E+02, confidence 1.572E-01, row_norm 9.851E-01}
[06/15/2024 02:26:48 gluefactory INFO] [E 26 | it 1500] loss {total 4.290E-01, last 5.586E-02, assignment_nll 5.586E-02, nll_pos 5.912E-02, nll_neg 5.260E-02, num_matchable 1.478E+02, num_unmatchable 3.482E+02, confidence 1.635E-01, row_norm 9.803E-01}
[06/15/2024 02:32:44 gluefactory INFO] [Validation] {match_recall 9.700E-01, match_precision 8.507E-01, accuracy 9.724E-01, average_precision 8.264E-01, loss/total 1.073E-01, loss/last 1.073E-01, loss/assignment_nll 1.073E-01, loss/nll_pos 1.467E-01, loss/nll_neg 6.779E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.744E-01}
[06/15/2024 02:40:09 gluefactory INFO] [Validation] {match_recall 9.710E-01, match_precision 8.539E-01, accuracy 9.731E-01, average_precision 8.297E-01, loss/total 1.023E-01, loss/last 1.023E-01, loss/assignment_nll 1.023E-01, loss/nll_pos 1.410E-01, loss/nll_neg 6.360E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.760E-01}
[06/15/2024 02:40:11 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_26_41714.tar
[06/15/2024 02:40:11 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_19_30000.tar
[06/15/2024 02:40:11 gluefactory INFO] Starting epoch 27
[06/15/2024 02:40:11 gluefactory INFO] lr changed from 1.9952623149688793e-05 to 1.584893192461113e-05
[06/15/2024 02:40:16 gluefactory INFO] [E 27 | it 0] loss {total 4.782E-01, last 8.823E-02, assignment_nll 8.823E-02, nll_pos 8.708E-02, nll_neg 8.939E-02, num_matchable 1.260E+02, num_unmatchable 3.724E+02, confidence 1.589E-01, row_norm 9.648E-01}
[06/15/2024 02:43:38 gluefactory INFO] [E 27 | it 100] loss {total 4.442E-01, last 6.232E-02, assignment_nll 6.232E-02, nll_pos 6.778E-02, nll_neg 5.685E-02, num_matchable 1.297E+02, num_unmatchable 3.675E+02, confidence 1.584E-01, row_norm 9.813E-01}
[06/15/2024 02:47:01 gluefactory INFO] [E 27 | it 200] loss {total 4.428E-01, last 5.238E-02, assignment_nll 5.238E-02, nll_pos 5.362E-02, nll_neg 5.113E-02, num_matchable 1.310E+02, num_unmatchable 3.636E+02, confidence 1.601E-01, row_norm 9.809E-01}
[06/15/2024 02:50:24 gluefactory INFO] [E 27 | it 300] loss {total 4.677E-01, last 6.419E-02, assignment_nll 6.419E-02, nll_pos 7.127E-02, nll_neg 5.711E-02, num_matchable 1.220E+02, num_unmatchable 3.766E+02, confidence 1.534E-01, row_norm 9.847E-01}
[06/15/2024 02:53:47 gluefactory INFO] [E 27 | it 400] loss {total 5.160E-01, last 9.520E-02, assignment_nll 9.520E-02, nll_pos 1.275E-01, nll_neg 6.294E-02, num_matchable 1.224E+02, num_unmatchable 3.757E+02, confidence 1.632E-01, row_norm 9.793E-01}
[06/15/2024 02:57:10 gluefactory INFO] [E 27 | it 500] loss {total 4.782E-01, last 7.940E-02, assignment_nll 7.940E-02, nll_pos 1.063E-01, nll_neg 5.251E-02, num_matchable 1.213E+02, num_unmatchable 3.768E+02, confidence 1.520E-01, row_norm 9.772E-01}
[06/15/2024 03:03:05 gluefactory INFO] [Validation] {match_recall 9.720E-01, match_precision 8.578E-01, accuracy 9.740E-01, average_precision 8.337E-01, loss/total 9.942E-02, loss/last 9.942E-02, loss/assignment_nll 9.942E-02, loss/nll_pos 1.384E-01, loss/nll_neg 6.045E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.783E-01}
[06/15/2024 03:03:05 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 03:03:06 gluefactory INFO] New best val: loss/total=0.09942490315153829
[06/15/2024 03:06:30 gluefactory INFO] [E 27 | it 600] loss {total 4.508E-01, last 7.324E-02, assignment_nll 7.324E-02, nll_pos 9.249E-02, nll_neg 5.399E-02, num_matchable 1.117E+02, num_unmatchable 3.836E+02, confidence 1.583E-01, row_norm 9.835E-01}
[06/15/2024 03:09:53 gluefactory INFO] [E 27 | it 700] loss {total 4.224E-01, last 5.514E-02, assignment_nll 5.514E-02, nll_pos 5.940E-02, nll_neg 5.087E-02, num_matchable 1.198E+02, num_unmatchable 3.787E+02, confidence 1.420E-01, row_norm 9.832E-01}
[06/15/2024 03:13:16 gluefactory INFO] [E 27 | it 800] loss {total 4.393E-01, last 5.090E-02, assignment_nll 5.090E-02, nll_pos 5.010E-02, nll_neg 5.170E-02, num_matchable 1.252E+02, num_unmatchable 3.728E+02, confidence 1.548E-01, row_norm 9.825E-01}
[06/15/2024 03:16:39 gluefactory INFO] [E 27 | it 900] loss {total 4.631E-01, last 7.414E-02, assignment_nll 7.414E-02, nll_pos 8.999E-02, nll_neg 5.829E-02, num_matchable 1.256E+02, num_unmatchable 3.732E+02, confidence 1.568E-01, row_norm 9.801E-01}
[06/15/2024 03:20:01 gluefactory INFO] [E 27 | it 1000] loss {total 4.284E-01, last 5.999E-02, assignment_nll 5.999E-02, nll_pos 6.723E-02, nll_neg 5.274E-02, num_matchable 1.392E+02, num_unmatchable 3.552E+02, confidence 1.575E-01, row_norm 9.814E-01}
[06/15/2024 03:25:57 gluefactory INFO] [Validation] {match_recall 9.716E-01, match_precision 8.529E-01, accuracy 9.728E-01, average_precision 8.288E-01, loss/total 1.028E-01, loss/last 1.028E-01, loss/assignment_nll 1.028E-01, loss/nll_pos 1.389E-01, loss/nll_neg 6.676E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.775E-01}
[06/15/2024 03:29:22 gluefactory INFO] [E 27 | it 1100] loss {total 4.694E-01, last 8.039E-02, assignment_nll 8.039E-02, nll_pos 1.040E-01, nll_neg 5.677E-02, num_matchable 1.253E+02, num_unmatchable 3.706E+02, confidence 1.597E-01, row_norm 9.775E-01}
[06/15/2024 03:32:45 gluefactory INFO] [E 27 | it 1200] loss {total 4.293E-01, last 5.772E-02, assignment_nll 5.772E-02, nll_pos 5.666E-02, nll_neg 5.878E-02, num_matchable 1.228E+02, num_unmatchable 3.742E+02, confidence 1.522E-01, row_norm 9.845E-01}
[06/15/2024 03:36:07 gluefactory INFO] [E 27 | it 1300] loss {total 5.085E-01, last 8.705E-02, assignment_nll 8.705E-02, nll_pos 1.119E-01, nll_neg 6.223E-02, num_matchable 1.318E+02, num_unmatchable 3.632E+02, confidence 1.677E-01, row_norm 9.711E-01}
[06/15/2024 03:39:30 gluefactory INFO] [E 27 | it 1400] loss {total 4.340E-01, last 6.111E-02, assignment_nll 6.111E-02, nll_pos 7.160E-02, nll_neg 5.061E-02, num_matchable 1.453E+02, num_unmatchable 3.534E+02, confidence 1.513E-01, row_norm 9.838E-01}
[06/15/2024 03:42:53 gluefactory INFO] [E 27 | it 1500] loss {total 4.221E-01, last 5.419E-02, assignment_nll 5.419E-02, nll_pos 5.741E-02, nll_neg 5.097E-02, num_matchable 1.425E+02, num_unmatchable 3.540E+02, confidence 1.600E-01, row_norm 9.822E-01}
[06/15/2024 03:48:48 gluefactory INFO] [Validation] {match_recall 9.721E-01, match_precision 8.598E-01, accuracy 9.746E-01, average_precision 8.357E-01, loss/total 9.659E-02, loss/last 9.659E-02, loss/assignment_nll 9.659E-02, loss/nll_pos 1.342E-01, loss/nll_neg 5.895E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.778E-01}
[06/15/2024 03:48:48 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 03:48:48 gluefactory INFO] New best val: loss/total=0.09658813099543716
[06/15/2024 03:56:14 gluefactory INFO] [Validation] {match_recall 9.722E-01, match_precision 8.564E-01, accuracy 9.738E-01, average_precision 8.325E-01, loss/total 9.781E-02, loss/last 9.781E-02, loss/assignment_nll 9.781E-02, loss/nll_pos 1.341E-01, loss/nll_neg 6.155E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.768E-01}
[06/15/2024 03:56:15 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_27_43259.tar
[06/15/2024 03:56:16 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_19_30899.tar
[06/15/2024 03:56:16 gluefactory INFO] Starting epoch 28
[06/15/2024 03:56:16 gluefactory INFO] lr changed from 1.584893192461113e-05 to 1.258925411794167e-05
[06/15/2024 03:56:21 gluefactory INFO] [E 28 | it 0] loss {total 6.056E-01, last 1.877E-01, assignment_nll 1.877E-01, nll_pos 3.049E-01, nll_neg 7.056E-02, num_matchable 1.235E+02, num_unmatchable 3.754E+02, confidence 1.537E-01, row_norm 9.800E-01}
[06/15/2024 03:59:43 gluefactory INFO] [E 28 | it 100] loss {total 4.630E-01, last 7.880E-02, assignment_nll 7.880E-02, nll_pos 1.048E-01, nll_neg 5.278E-02, num_matchable 1.408E+02, num_unmatchable 3.567E+02, confidence 1.599E-01, row_norm 9.804E-01}
[06/15/2024 04:03:06 gluefactory INFO] [E 28 | it 200] loss {total 3.876E-01, last 5.191E-02, assignment_nll 5.191E-02, nll_pos 5.811E-02, nll_neg 4.571E-02, num_matchable 1.368E+02, num_unmatchable 3.588E+02, confidence 1.460E-01, row_norm 9.845E-01}
[06/15/2024 04:06:29 gluefactory INFO] [E 28 | it 300] loss {total 4.788E-01, last 6.933E-02, assignment_nll 6.933E-02, nll_pos 7.695E-02, nll_neg 6.170E-02, num_matchable 1.163E+02, num_unmatchable 3.827E+02, confidence 1.464E-01, row_norm 9.773E-01}
[06/15/2024 04:09:51 gluefactory INFO] [E 28 | it 400] loss {total 4.512E-01, last 5.981E-02, assignment_nll 5.981E-02, nll_pos 6.578E-02, nll_neg 5.384E-02, num_matchable 1.233E+02, num_unmatchable 3.717E+02, confidence 1.571E-01, row_norm 9.766E-01}
[06/15/2024 04:13:14 gluefactory INFO] [E 28 | it 500] loss {total 4.199E-01, last 5.468E-02, assignment_nll 5.468E-02, nll_pos 5.488E-02, nll_neg 5.447E-02, num_matchable 1.337E+02, num_unmatchable 3.630E+02, confidence 1.585E-01, row_norm 9.822E-01}
[06/15/2024 04:19:09 gluefactory INFO] [Validation] {match_recall 9.726E-01, match_precision 8.546E-01, accuracy 9.734E-01, average_precision 8.313E-01, loss/total 9.646E-02, loss/last 9.646E-02, loss/assignment_nll 9.646E-02, loss/nll_pos 1.297E-01, loss/nll_neg 6.322E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.775E-01}
[06/15/2024 04:19:09 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 04:19:10 gluefactory INFO] New best val: loss/total=0.09645858406297388
[06/15/2024 04:22:34 gluefactory INFO] [E 28 | it 600] loss {total 4.345E-01, last 6.326E-02, assignment_nll 6.326E-02, nll_pos 7.769E-02, nll_neg 4.882E-02, num_matchable 1.126E+02, num_unmatchable 3.841E+02, confidence 1.507E-01, row_norm 9.872E-01}
[06/15/2024 04:25:57 gluefactory INFO] [E 28 | it 700] loss {total 3.941E-01, last 4.582E-02, assignment_nll 4.582E-02, nll_pos 4.091E-02, nll_neg 5.073E-02, num_matchable 1.278E+02, num_unmatchable 3.713E+02, confidence 1.454E-01, row_norm 9.860E-01}
[06/15/2024 04:29:20 gluefactory INFO] [E 28 | it 800] loss {total 5.080E-01, last 6.450E-02, assignment_nll 6.450E-02, nll_pos 6.219E-02, nll_neg 6.681E-02, num_matchable 1.178E+02, num_unmatchable 3.777E+02, confidence 1.680E-01, row_norm 9.810E-01}
[06/15/2024 04:32:42 gluefactory INFO] [E 28 | it 900] loss {total 4.325E-01, last 6.060E-02, assignment_nll 6.060E-02, nll_pos 7.028E-02, nll_neg 5.093E-02, num_matchable 1.278E+02, num_unmatchable 3.714E+02, confidence 1.568E-01, row_norm 9.841E-01}
[06/15/2024 04:36:05 gluefactory INFO] [E 28 | it 1000] loss {total 4.680E-01, last 1.143E-01, assignment_nll 1.143E-01, nll_pos 1.626E-01, nll_neg 6.601E-02, num_matchable 1.431E+02, num_unmatchable 3.531E+02, confidence 1.548E-01, row_norm 9.754E-01}
[06/15/2024 04:42:01 gluefactory INFO] [Validation] {match_recall 9.728E-01, match_precision 8.566E-01, accuracy 9.738E-01, average_precision 8.330E-01, loss/total 9.634E-02, loss/last 9.634E-02, loss/assignment_nll 9.634E-02, loss/nll_pos 1.295E-01, loss/nll_neg 6.313E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.773E-01}
[06/15/2024 04:42:01 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 04:42:01 gluefactory INFO] New best val: loss/total=0.0963368165900886
[06/15/2024 04:45:26 gluefactory INFO] [E 28 | it 1100] loss {total 4.612E-01, last 8.443E-02, assignment_nll 8.443E-02, nll_pos 1.111E-01, nll_neg 5.773E-02, num_matchable 1.307E+02, num_unmatchable 3.692E+02, confidence 1.545E-01, row_norm 9.801E-01}
[06/15/2024 04:48:48 gluefactory INFO] [E 28 | it 1200] loss {total 4.477E-01, last 8.043E-02, assignment_nll 8.043E-02, nll_pos 1.074E-01, nll_neg 5.345E-02, num_matchable 1.221E+02, num_unmatchable 3.743E+02, confidence 1.489E-01, row_norm 9.788E-01}
[06/15/2024 04:52:11 gluefactory INFO] [E 28 | it 1300] loss {total 4.843E-01, last 6.606E-02, assignment_nll 6.606E-02, nll_pos 7.138E-02, nll_neg 6.075E-02, num_matchable 1.377E+02, num_unmatchable 3.570E+02, confidence 1.673E-01, row_norm 9.788E-01}
[06/15/2024 04:55:34 gluefactory INFO] [E 28 | it 1400] loss {total 4.418E-01, last 7.735E-02, assignment_nll 7.735E-02, nll_pos 6.752E-02, nll_neg 8.718E-02, num_matchable 1.295E+02, num_unmatchable 3.700E+02, confidence 1.465E-01, row_norm 9.661E-01}
[06/15/2024 04:58:57 gluefactory INFO] [E 28 | it 1500] loss {total 4.868E-01, last 7.817E-02, assignment_nll 7.817E-02, nll_pos 9.481E-02, nll_neg 6.152E-02, num_matchable 1.414E+02, num_unmatchable 3.541E+02, confidence 1.627E-01, row_norm 9.814E-01}
[06/15/2024 05:04:51 gluefactory INFO] [Validation] {match_recall 9.727E-01, match_precision 8.621E-01, accuracy 9.752E-01, average_precision 8.383E-01, loss/total 9.535E-02, loss/last 9.535E-02, loss/assignment_nll 9.535E-02, loss/nll_pos 1.326E-01, loss/nll_neg 5.814E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.784E-01}
[06/15/2024 05:04:51 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 05:04:52 gluefactory INFO] New best val: loss/total=0.09535457182883214
[06/15/2024 05:12:17 gluefactory INFO] [Validation] {match_recall 9.726E-01, match_precision 8.598E-01, accuracy 9.747E-01, average_precision 8.361E-01, loss/total 9.533E-02, loss/last 9.533E-02, loss/assignment_nll 9.533E-02, loss/nll_pos 1.321E-01, loss/nll_neg 5.852E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.774E-01}
[06/15/2024 05:12:17 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 05:12:17 gluefactory INFO] New best val: loss/total=0.0953315245452402
[06/15/2024 05:12:19 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_28_44804.tar
[06/15/2024 05:12:20 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_20_32444.tar
[06/15/2024 05:12:20 gluefactory INFO] Starting epoch 29
[06/15/2024 05:12:20 gluefactory INFO] lr changed from 1.258925411794167e-05 to 9.999999999999997e-06
[06/15/2024 05:12:25 gluefactory INFO] [E 29 | it 0] loss {total 5.016E-01, last 7.963E-02, assignment_nll 7.963E-02, nll_pos 1.025E-01, nll_neg 5.672E-02, num_matchable 1.207E+02, num_unmatchable 3.787E+02, confidence 1.602E-01, row_norm 9.802E-01}
[06/15/2024 05:15:47 gluefactory INFO] [E 29 | it 100] loss {total 4.493E-01, last 6.750E-02, assignment_nll 6.750E-02, nll_pos 7.883E-02, nll_neg 5.617E-02, num_matchable 1.230E+02, num_unmatchable 3.756E+02, confidence 1.501E-01, row_norm 9.813E-01}
[06/15/2024 05:18:59 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_29_45000.tar
[06/15/2024 05:19:00 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_21_33989.tar
[06/15/2024 05:19:10 gluefactory INFO] [E 29 | it 200] loss {total 4.377E-01, last 5.288E-02, assignment_nll 5.288E-02, nll_pos 5.579E-02, nll_neg 4.996E-02, num_matchable 1.313E+02, num_unmatchable 3.623E+02, confidence 1.574E-01, row_norm 9.837E-01}
[06/15/2024 05:22:33 gluefactory INFO] [E 29 | it 300] loss {total 4.583E-01, last 5.382E-02, assignment_nll 5.382E-02, nll_pos 5.385E-02, nll_neg 5.380E-02, num_matchable 1.306E+02, num_unmatchable 3.675E+02, confidence 1.589E-01, row_norm 9.828E-01}
[06/15/2024 05:25:55 gluefactory INFO] [E 29 | it 400] loss {total 5.120E-01, last 8.068E-02, assignment_nll 8.068E-02, nll_pos 1.041E-01, nll_neg 5.729E-02, num_matchable 1.131E+02, num_unmatchable 3.840E+02, confidence 1.560E-01, row_norm 9.803E-01}
[06/15/2024 05:29:18 gluefactory INFO] [E 29 | it 500] loss {total 4.198E-01, last 4.892E-02, assignment_nll 4.892E-02, nll_pos 4.904E-02, nll_neg 4.880E-02, num_matchable 1.339E+02, num_unmatchable 3.639E+02, confidence 1.591E-01, row_norm 9.823E-01}
[06/15/2024 05:35:14 gluefactory INFO] [Validation] {match_recall 9.719E-01, match_precision 8.556E-01, accuracy 9.738E-01, average_precision 8.322E-01, loss/total 9.924E-02, loss/last 9.924E-02, loss/assignment_nll 9.924E-02, loss/nll_pos 1.363E-01, loss/nll_neg 6.214E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.759E-01}
[06/15/2024 05:38:38 gluefactory INFO] [E 29 | it 600] loss {total 4.704E-01, last 5.864E-02, assignment_nll 5.864E-02, nll_pos 6.513E-02, nll_neg 5.214E-02, num_matchable 1.227E+02, num_unmatchable 3.714E+02, confidence 1.655E-01, row_norm 9.850E-01}
[06/15/2024 05:42:00 gluefactory INFO] [E 29 | it 700] loss {total 4.748E-01, last 7.438E-02, assignment_nll 7.438E-02, nll_pos 9.573E-02, nll_neg 5.303E-02, num_matchable 1.180E+02, num_unmatchable 3.797E+02, confidence 1.498E-01, row_norm 9.841E-01}
[06/15/2024 05:45:23 gluefactory INFO] [E 29 | it 800] loss {total 4.413E-01, last 5.437E-02, assignment_nll 5.437E-02, nll_pos 5.339E-02, nll_neg 5.535E-02, num_matchable 1.241E+02, num_unmatchable 3.737E+02, confidence 1.582E-01, row_norm 9.844E-01}
[06/15/2024 05:48:46 gluefactory INFO] [E 29 | it 900] loss {total 5.109E-01, last 1.170E-01, assignment_nll 1.170E-01, nll_pos 1.780E-01, nll_neg 5.608E-02, num_matchable 1.355E+02, num_unmatchable 3.624E+02, confidence 1.610E-01, row_norm 9.767E-01}
[06/15/2024 05:52:09 gluefactory INFO] [E 29 | it 1000] loss {total 4.003E-01, last 5.103E-02, assignment_nll 5.103E-02, nll_pos 5.985E-02, nll_neg 4.220E-02, num_matchable 1.489E+02, num_unmatchable 3.483E+02, confidence 1.518E-01, row_norm 9.842E-01}
[06/15/2024 05:58:04 gluefactory INFO] [Validation] {match_recall 9.731E-01, match_precision 8.609E-01, accuracy 9.749E-01, average_precision 8.375E-01, loss/total 9.329E-02, loss/last 9.329E-02, loss/assignment_nll 9.329E-02, loss/nll_pos 1.281E-01, loss/nll_neg 5.844E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.784E-01}
[06/15/2024 05:58:04 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 05:58:04 gluefactory INFO] New best val: loss/total=0.0932900456175742
[06/15/2024 06:01:28 gluefactory INFO] [E 29 | it 1100] loss {total 4.438E-01, last 7.085E-02, assignment_nll 7.085E-02, nll_pos 8.560E-02, nll_neg 5.610E-02, num_matchable 1.270E+02, num_unmatchable 3.714E+02, confidence 1.520E-01, row_norm 9.835E-01}
[06/15/2024 06:04:51 gluefactory INFO] [E 29 | it 1200] loss {total 4.282E-01, last 5.777E-02, assignment_nll 5.777E-02, nll_pos 6.401E-02, nll_neg 5.154E-02, num_matchable 1.251E+02, num_unmatchable 3.711E+02, confidence 1.548E-01, row_norm 9.843E-01}
[06/15/2024 06:08:14 gluefactory INFO] [E 29 | it 1300] loss {total 4.853E-01, last 7.728E-02, assignment_nll 7.728E-02, nll_pos 9.895E-02, nll_neg 5.562E-02, num_matchable 1.157E+02, num_unmatchable 3.818E+02, confidence 1.531E-01, row_norm 9.783E-01}
[06/15/2024 06:11:37 gluefactory INFO] [E 29 | it 1400] loss {total 4.193E-01, last 6.944E-02, assignment_nll 6.944E-02, nll_pos 8.680E-02, nll_neg 5.207E-02, num_matchable 1.402E+02, num_unmatchable 3.577E+02, confidence 1.482E-01, row_norm 9.810E-01}
[06/15/2024 06:14:59 gluefactory INFO] [E 29 | it 1500] loss {total 4.282E-01, last 5.563E-02, assignment_nll 5.563E-02, nll_pos 6.360E-02, nll_neg 4.765E-02, num_matchable 1.523E+02, num_unmatchable 3.409E+02, confidence 1.624E-01, row_norm 9.793E-01}
[06/15/2024 06:20:55 gluefactory INFO] [Validation] {match_recall 9.730E-01, match_precision 8.596E-01, accuracy 9.746E-01, average_precision 8.362E-01, loss/total 9.398E-02, loss/last 9.398E-02, loss/assignment_nll 9.398E-02, loss/nll_pos 1.276E-01, loss/nll_neg 6.035E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.780E-01}
[06/15/2024 06:28:19 gluefactory INFO] [Validation] {match_recall 9.731E-01, match_precision 8.606E-01, accuracy 9.749E-01, average_precision 8.373E-01, loss/total 9.376E-02, loss/last 9.376E-02, loss/assignment_nll 9.376E-02, loss/nll_pos 1.287E-01, loss/nll_neg 5.879E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.778E-01}
[06/15/2024 06:28:21 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_29_46349.tar
[06/15/2024 06:28:22 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_22_35000.tar
[06/15/2024 06:28:22 gluefactory INFO] Starting epoch 30
[06/15/2024 06:28:22 gluefactory INFO] lr changed from 9.999999999999997e-06 to 7.943282347242813e-06
[06/15/2024 06:28:28 gluefactory INFO] [E 30 | it 0] loss {total 4.451E-01, last 6.979E-02, assignment_nll 6.979E-02, nll_pos 8.682E-02, nll_neg 5.276E-02, num_matchable 1.232E+02, num_unmatchable 3.756E+02, confidence 1.533E-01, row_norm 9.861E-01}
[06/15/2024 06:31:49 gluefactory INFO] [E 30 | it 100] loss {total 4.430E-01, last 6.726E-02, assignment_nll 6.726E-02, nll_pos 8.031E-02, nll_neg 5.421E-02, num_matchable 1.334E+02, num_unmatchable 3.635E+02, confidence 1.564E-01, row_norm 9.832E-01}
[06/15/2024 06:35:10 gluefactory INFO] [E 30 | it 200] loss {total 4.635E-01, last 6.842E-02, assignment_nll 6.842E-02, nll_pos 8.499E-02, nll_neg 5.185E-02, num_matchable 1.262E+02, num_unmatchable 3.686E+02, confidence 1.509E-01, row_norm 9.827E-01}
[06/15/2024 06:38:31 gluefactory INFO] [E 30 | it 300] loss {total 5.918E-01, last 1.818E-01, assignment_nll 1.818E-01, nll_pos 2.758E-01, nll_neg 8.786E-02, num_matchable 1.181E+02, num_unmatchable 3.785E+02, confidence 1.550E-01, row_norm 9.632E-01}
[06/15/2024 06:41:53 gluefactory INFO] [E 30 | it 400] loss {total 5.969E-01, last 1.290E-01, assignment_nll 1.290E-01, nll_pos 1.804E-01, nll_neg 7.771E-02, num_matchable 1.063E+02, num_unmatchable 3.914E+02, confidence 1.538E-01, row_norm 9.674E-01}
[06/15/2024 06:45:14 gluefactory INFO] [E 30 | it 500] loss {total 4.509E-01, last 6.152E-02, assignment_nll 6.152E-02, nll_pos 6.711E-02, nll_neg 5.593E-02, num_matchable 1.189E+02, num_unmatchable 3.800E+02, confidence 1.507E-01, row_norm 9.837E-01}
[06/15/2024 06:51:08 gluefactory INFO] [Validation] {match_recall 9.732E-01, match_precision 8.628E-01, accuracy 9.755E-01, average_precision 8.394E-01, loss/total 9.285E-02, loss/last 9.285E-02, loss/assignment_nll 9.285E-02, loss/nll_pos 1.284E-01, loss/nll_neg 5.725E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.784E-01}
[06/15/2024 06:51:08 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 06:51:08 gluefactory INFO] New best val: loss/total=0.09284509881013499
[06/15/2024 06:54:31 gluefactory INFO] [E 30 | it 600] loss {total 4.238E-01, last 5.275E-02, assignment_nll 5.275E-02, nll_pos 5.769E-02, nll_neg 4.781E-02, num_matchable 1.287E+02, num_unmatchable 3.673E+02, confidence 1.577E-01, row_norm 9.873E-01}
[06/15/2024 06:57:52 gluefactory INFO] [E 30 | it 700] loss {total 4.385E-01, last 6.258E-02, assignment_nll 6.258E-02, nll_pos 7.823E-02, nll_neg 4.693E-02, num_matchable 1.198E+02, num_unmatchable 3.789E+02, confidence 1.498E-01, row_norm 9.862E-01}
[06/15/2024 07:01:14 gluefactory INFO] [E 30 | it 800] loss {total 5.409E-01, last 1.768E-01, assignment_nll 1.768E-01, nll_pos 2.807E-01, nll_neg 7.290E-02, num_matchable 1.200E+02, num_unmatchable 3.776E+02, confidence 1.509E-01, row_norm 9.710E-01}
[06/15/2024 07:04:35 gluefactory INFO] [E 30 | it 900] loss {total 4.377E-01, last 5.351E-02, assignment_nll 5.351E-02, nll_pos 5.314E-02, nll_neg 5.388E-02, num_matchable 1.150E+02, num_unmatchable 3.845E+02, confidence 1.486E-01, row_norm 9.845E-01}
[06/15/2024 07:07:56 gluefactory INFO] [E 30 | it 1000] loss {total 4.828E-01, last 1.591E-01, assignment_nll 1.591E-01, nll_pos 2.544E-01, nll_neg 6.370E-02, num_matchable 1.496E+02, num_unmatchable 3.482E+02, confidence 1.405E-01, row_norm 9.680E-01}
[06/15/2024 07:13:50 gluefactory INFO] [Validation] {match_recall 9.733E-01, match_precision 8.625E-01, accuracy 9.753E-01, average_precision 8.390E-01, loss/total 9.366E-02, loss/last 9.366E-02, loss/assignment_nll 9.366E-02, loss/nll_pos 1.285E-01, loss/nll_neg 5.882E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.787E-01}
[06/15/2024 07:17:13 gluefactory INFO] [E 30 | it 1100] loss {total 4.385E-01, last 7.419E-02, assignment_nll 7.419E-02, nll_pos 9.297E-02, nll_neg 5.540E-02, num_matchable 1.453E+02, num_unmatchable 3.507E+02, confidence 1.612E-01, row_norm 9.765E-01}
[06/15/2024 07:20:34 gluefactory INFO] [E 30 | it 1200] loss {total 3.856E-01, last 4.614E-02, assignment_nll 4.614E-02, nll_pos 5.195E-02, nll_neg 4.033E-02, num_matchable 1.241E+02, num_unmatchable 3.731E+02, confidence 1.462E-01, row_norm 9.862E-01}
[06/15/2024 07:23:55 gluefactory INFO] [E 30 | it 1300] loss {total 4.339E-01, last 6.422E-02, assignment_nll 6.422E-02, nll_pos 7.678E-02, nll_neg 5.165E-02, num_matchable 1.241E+02, num_unmatchable 3.727E+02, confidence 1.525E-01, row_norm 9.839E-01}
[06/15/2024 07:27:17 gluefactory INFO] [E 30 | it 1400] loss {total 4.112E-01, last 8.359E-02, assignment_nll 8.359E-02, nll_pos 1.165E-01, nll_neg 5.067E-02, num_matchable 1.335E+02, num_unmatchable 3.683E+02, confidence 1.381E-01, row_norm 9.877E-01}
[06/15/2024 07:30:38 gluefactory INFO] [E 30 | it 1500] loss {total 4.393E-01, last 6.341E-02, assignment_nll 6.341E-02, nll_pos 8.022E-02, nll_neg 4.661E-02, num_matchable 1.412E+02, num_unmatchable 3.519E+02, confidence 1.644E-01, row_norm 9.762E-01}
[06/15/2024 07:36:32 gluefactory INFO] [Validation] {match_recall 9.735E-01, match_precision 8.622E-01, accuracy 9.753E-01, average_precision 8.389E-01, loss/total 9.218E-02, loss/last 9.218E-02, loss/assignment_nll 9.218E-02, loss/nll_pos 1.268E-01, loss/nll_neg 5.760E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.784E-01}
[06/15/2024 07:36:32 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 07:36:33 gluefactory INFO] New best val: loss/total=0.09217890966655
[06/15/2024 07:43:58 gluefactory INFO] [Validation] {match_recall 9.735E-01, match_precision 8.613E-01, accuracy 9.751E-01, average_precision 8.381E-01, loss/total 9.225E-02, loss/last 9.225E-02, loss/assignment_nll 9.225E-02, loss/nll_pos 1.258E-01, loss/nll_neg 5.869E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.781E-01}
[06/15/2024 07:43:59 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_30_47894.tar
[06/15/2024 07:44:00 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_22_35534.tar
[06/15/2024 07:44:00 gluefactory INFO] Starting epoch 31
[06/15/2024 07:44:00 gluefactory INFO] lr changed from 7.943282347242813e-06 to 6.309573444801931e-06
[06/15/2024 07:44:05 gluefactory INFO] [E 31 | it 0] loss {total 4.987E-01, last 7.832E-02, assignment_nll 7.832E-02, nll_pos 9.144E-02, nll_neg 6.521E-02, num_matchable 1.260E+02, num_unmatchable 3.708E+02, confidence 1.635E-01, row_norm 9.805E-01}
[06/15/2024 07:47:27 gluefactory INFO] [E 31 | it 100] loss {total 5.041E-01, last 1.228E-01, assignment_nll 1.228E-01, nll_pos 1.747E-01, nll_neg 7.086E-02, num_matchable 1.307E+02, num_unmatchable 3.677E+02, confidence 1.540E-01, row_norm 9.683E-01}
[06/15/2024 07:50:48 gluefactory INFO] [E 31 | it 200] loss {total 4.591E-01, last 5.786E-02, assignment_nll 5.786E-02, nll_pos 6.058E-02, nll_neg 5.515E-02, num_matchable 1.326E+02, num_unmatchable 3.625E+02, confidence 1.591E-01, row_norm 9.809E-01}
[06/15/2024 07:54:09 gluefactory INFO] [E 31 | it 300] loss {total 5.178E-01, last 8.005E-02, assignment_nll 8.005E-02, nll_pos 9.595E-02, nll_neg 6.415E-02, num_matchable 1.208E+02, num_unmatchable 3.772E+02, confidence 1.547E-01, row_norm 9.794E-01}
[06/15/2024 07:57:31 gluefactory INFO] [E 31 | it 400] loss {total 5.967E-01, last 1.233E-01, assignment_nll 1.233E-01, nll_pos 1.774E-01, nll_neg 6.933E-02, num_matchable 1.091E+02, num_unmatchable 3.878E+02, confidence 1.555E-01, row_norm 9.667E-01}
[06/15/2024 08:00:52 gluefactory INFO] [E 31 | it 500] loss {total 4.580E-01, last 8.028E-02, assignment_nll 8.028E-02, nll_pos 1.064E-01, nll_neg 5.418E-02, num_matchable 1.221E+02, num_unmatchable 3.754E+02, confidence 1.483E-01, row_norm 9.807E-01}
[06/15/2024 08:06:46 gluefactory INFO] [Validation] {match_recall 9.733E-01, match_precision 8.648E-01, accuracy 9.759E-01, average_precision 8.413E-01, loss/total 9.237E-02, loss/last 9.237E-02, loss/assignment_nll 9.237E-02, loss/nll_pos 1.292E-01, loss/nll_neg 5.556E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.786E-01}
[06/15/2024 08:10:09 gluefactory INFO] [E 31 | it 600] loss {total 4.484E-01, last 7.192E-02, assignment_nll 7.192E-02, nll_pos 9.136E-02, nll_neg 5.249E-02, num_matchable 1.247E+02, num_unmatchable 3.709E+02, confidence 1.558E-01, row_norm 9.825E-01}
[06/15/2024 08:13:30 gluefactory INFO] [E 31 | it 700] loss {total 4.471E-01, last 6.861E-02, assignment_nll 6.861E-02, nll_pos 8.117E-02, nll_neg 5.606E-02, num_matchable 1.288E+02, num_unmatchable 3.698E+02, confidence 1.455E-01, row_norm 9.817E-01}
[06/15/2024 08:16:51 gluefactory INFO] [E 31 | it 800] loss {total 4.454E-01, last 5.977E-02, assignment_nll 5.977E-02, nll_pos 6.469E-02, nll_neg 5.486E-02, num_matchable 1.266E+02, num_unmatchable 3.697E+02, confidence 1.582E-01, row_norm 9.829E-01}
[06/15/2024 08:20:13 gluefactory INFO] [E 31 | it 900] loss {total 4.189E-01, last 5.233E-02, assignment_nll 5.233E-02, nll_pos 5.256E-02, nll_neg 5.210E-02, num_matchable 1.233E+02, num_unmatchable 3.754E+02, confidence 1.542E-01, row_norm 9.835E-01}
[06/15/2024 08:23:34 gluefactory INFO] [E 31 | it 1000] loss {total 4.082E-01, last 5.263E-02, assignment_nll 5.263E-02, nll_pos 5.814E-02, nll_neg 4.713E-02, num_matchable 1.441E+02, num_unmatchable 3.517E+02, confidence 1.529E-01, row_norm 9.830E-01}
[06/15/2024 08:29:28 gluefactory INFO] [Validation] {match_recall 9.736E-01, match_precision 8.606E-01, accuracy 9.749E-01, average_precision 8.376E-01, loss/total 9.226E-02, loss/last 9.226E-02, loss/assignment_nll 9.226E-02, loss/nll_pos 1.240E-01, loss/nll_neg 6.056E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.785E-01}
[06/15/2024 08:32:50 gluefactory INFO] [E 31 | it 1100] loss {total 4.826E-01, last 6.320E-02, assignment_nll 6.320E-02, nll_pos 7.057E-02, nll_neg 5.582E-02, num_matchable 1.266E+02, num_unmatchable 3.717E+02, confidence 1.641E-01, row_norm 9.820E-01}
[06/15/2024 08:36:12 gluefactory INFO] [E 31 | it 1200] loss {total 4.513E-01, last 6.768E-02, assignment_nll 6.768E-02, nll_pos 8.396E-02, nll_neg 5.139E-02, num_matchable 1.337E+02, num_unmatchable 3.617E+02, confidence 1.593E-01, row_norm 9.794E-01}
[06/15/2024 08:39:33 gluefactory INFO] [E 31 | it 1300] loss {total 4.705E-01, last 6.350E-02, assignment_nll 6.350E-02, nll_pos 7.616E-02, nll_neg 5.085E-02, num_matchable 1.310E+02, num_unmatchable 3.649E+02, confidence 1.630E-01, row_norm 9.795E-01}
[06/15/2024 08:42:54 gluefactory INFO] [E 31 | it 1400] loss {total 4.478E-01, last 7.090E-02, assignment_nll 7.090E-02, nll_pos 9.092E-02, nll_neg 5.088E-02, num_matchable 1.362E+02, num_unmatchable 3.633E+02, confidence 1.473E-01, row_norm 9.839E-01}
[06/15/2024 08:46:16 gluefactory INFO] [E 31 | it 1500] loss {total 4.672E-01, last 8.692E-02, assignment_nll 8.692E-02, nll_pos 1.192E-01, nll_neg 5.466E-02, num_matchable 1.347E+02, num_unmatchable 3.612E+02, confidence 1.494E-01, row_norm 9.806E-01}
[06/15/2024 08:52:10 gluefactory INFO] [Validation] {match_recall 9.738E-01, match_precision 8.602E-01, accuracy 9.748E-01, average_precision 8.372E-01, loss/total 9.187E-02, loss/last 9.187E-02, loss/assignment_nll 9.187E-02, loss/nll_pos 1.238E-01, loss/nll_neg 5.999E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.783E-01}
[06/15/2024 08:52:10 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 08:52:11 gluefactory INFO] New best val: loss/total=0.09187258376587674
[06/15/2024 08:59:36 gluefactory INFO] [Validation] {match_recall 9.736E-01, match_precision 8.638E-01, accuracy 9.757E-01, average_precision 8.406E-01, loss/total 9.168E-02, loss/last 9.168E-02, loss/assignment_nll 9.168E-02, loss/nll_pos 1.266E-01, loss/nll_neg 5.672E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.782E-01}
[06/15/2024 08:59:36 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 08:59:37 gluefactory INFO] New best val: loss/total=0.09168039740920597
[06/15/2024 08:59:38 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_31_49439.tar
[06/15/2024 08:59:39 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_23_37079.tar
[06/15/2024 08:59:39 gluefactory INFO] Starting epoch 32
[06/15/2024 08:59:39 gluefactory INFO] lr changed from 6.309573444801931e-06 to 5.011872336272722e-06
[06/15/2024 08:59:44 gluefactory INFO] [E 32 | it 0] loss {total 4.828E-01, last 6.141E-02, assignment_nll 6.141E-02, nll_pos 6.598E-02, nll_neg 5.685E-02, num_matchable 1.228E+02, num_unmatchable 3.756E+02, confidence 1.624E-01, row_norm 9.826E-01}
[06/15/2024 09:03:06 gluefactory INFO] [E 32 | it 100] loss {total 4.777E-01, last 1.007E-01, assignment_nll 1.007E-01, nll_pos 1.416E-01, nll_neg 5.978E-02, num_matchable 1.286E+02, num_unmatchable 3.690E+02, confidence 1.526E-01, row_norm 9.751E-01}
[06/15/2024 09:06:27 gluefactory INFO] [E 32 | it 200] loss {total 4.382E-01, last 5.913E-02, assignment_nll 5.913E-02, nll_pos 7.183E-02, nll_neg 4.643E-02, num_matchable 1.234E+02, num_unmatchable 3.705E+02, confidence 1.561E-01, row_norm 9.797E-01}
[06/15/2024 09:09:48 gluefactory INFO] [E 32 | it 300] loss {total 4.320E-01, last 5.043E-02, assignment_nll 5.043E-02, nll_pos 5.307E-02, nll_neg 4.778E-02, num_matchable 1.147E+02, num_unmatchable 3.846E+02, confidence 1.505E-01, row_norm 9.875E-01}
[06/15/2024 09:13:09 gluefactory INFO] [E 32 | it 400] loss {total 4.966E-01, last 7.586E-02, assignment_nll 7.586E-02, nll_pos 9.124E-02, nll_neg 6.048E-02, num_matchable 1.177E+02, num_unmatchable 3.785E+02, confidence 1.582E-01, row_norm 9.773E-01}
[06/15/2024 09:16:31 gluefactory INFO] [E 32 | it 500] loss {total 4.352E-01, last 5.200E-02, assignment_nll 5.200E-02, nll_pos 5.597E-02, nll_neg 4.802E-02, num_matchable 1.405E+02, num_unmatchable 3.569E+02, confidence 1.608E-01, row_norm 9.813E-01}
[06/15/2024 09:22:24 gluefactory INFO] [Validation] {match_recall 9.738E-01, match_precision 8.641E-01, accuracy 9.758E-01, average_precision 8.410E-01, loss/total 9.119E-02, loss/last 9.119E-02, loss/assignment_nll 9.119E-02, loss/nll_pos 1.253E-01, loss/nll_neg 5.708E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.786E-01}
[06/15/2024 09:22:24 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 09:22:25 gluefactory INFO] New best val: loss/total=0.09119488990857437
[06/15/2024 09:24:27 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_32_50000.tar
[06/15/2024 09:24:27 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_24_38624.tar
[06/15/2024 09:25:48 gluefactory INFO] [E 32 | it 600] loss {total 4.104E-01, last 5.486E-02, assignment_nll 5.486E-02, nll_pos 6.551E-02, nll_neg 4.421E-02, num_matchable 1.206E+02, num_unmatchable 3.751E+02, confidence 1.544E-01, row_norm 9.859E-01}
[06/15/2024 09:29:09 gluefactory INFO] [E 32 | it 700] loss {total 4.044E-01, last 4.460E-02, assignment_nll 4.460E-02, nll_pos 3.924E-02, nll_neg 4.995E-02, num_matchable 1.329E+02, num_unmatchable 3.653E+02, confidence 1.452E-01, row_norm 9.864E-01}
[06/15/2024 09:32:31 gluefactory INFO] [E 32 | it 800] loss {total 4.628E-01, last 7.552E-02, assignment_nll 7.552E-02, nll_pos 9.737E-02, nll_neg 5.367E-02, num_matchable 1.160E+02, num_unmatchable 3.820E+02, confidence 1.542E-01, row_norm 9.811E-01}
[06/15/2024 09:35:52 gluefactory INFO] [E 32 | it 900] loss {total 4.800E-01, last 6.088E-02, assignment_nll 6.088E-02, nll_pos 6.580E-02, nll_neg 5.596E-02, num_matchable 1.262E+02, num_unmatchable 3.717E+02, confidence 1.619E-01, row_norm 9.813E-01}
[06/15/2024 09:39:13 gluefactory INFO] [E 32 | it 1000] loss {total 4.953E-01, last 1.350E-01, assignment_nll 1.350E-01, nll_pos 2.146E-01, nll_neg 5.540E-02, num_matchable 1.398E+02, num_unmatchable 3.571E+02, confidence 1.505E-01, row_norm 9.806E-01}
[06/15/2024 09:45:07 gluefactory INFO] [Validation] {match_recall 9.739E-01, match_precision 8.643E-01, accuracy 9.758E-01, average_precision 8.411E-01, loss/total 9.060E-02, loss/last 9.060E-02, loss/assignment_nll 9.060E-02, loss/nll_pos 1.251E-01, loss/nll_neg 5.606E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.787E-01}
[06/15/2024 09:45:07 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 09:45:08 gluefactory INFO] New best val: loss/total=0.09059936799905084
[06/15/2024 09:48:31 gluefactory INFO] [E 32 | it 1100] loss {total 5.088E-01, last 9.348E-02, assignment_nll 9.348E-02, nll_pos 1.237E-01, nll_neg 6.324E-02, num_matchable 1.308E+02, num_unmatchable 3.668E+02, confidence 1.634E-01, row_norm 9.767E-01}
[06/15/2024 09:51:52 gluefactory INFO] [E 32 | it 1200] loss {total 4.987E-01, last 1.277E-01, assignment_nll 1.277E-01, nll_pos 1.896E-01, nll_neg 6.586E-02, num_matchable 1.311E+02, num_unmatchable 3.660E+02, confidence 1.475E-01, row_norm 9.704E-01}
[06/15/2024 09:55:13 gluefactory INFO] [E 32 | it 1300] loss {total 4.813E-01, last 8.886E-02, assignment_nll 8.886E-02, nll_pos 1.218E-01, nll_neg 5.597E-02, num_matchable 1.292E+02, num_unmatchable 3.666E+02, confidence 1.593E-01, row_norm 9.772E-01}
[06/15/2024 09:58:35 gluefactory INFO] [E 32 | it 1400] loss {total 4.326E-01, last 6.994E-02, assignment_nll 6.994E-02, nll_pos 8.929E-02, nll_neg 5.058E-02, num_matchable 1.412E+02, num_unmatchable 3.576E+02, confidence 1.479E-01, row_norm 9.826E-01}
[06/15/2024 10:01:57 gluefactory INFO] [E 32 | it 1500] loss {total 4.178E-01, last 4.871E-02, assignment_nll 4.871E-02, nll_pos 4.855E-02, nll_neg 4.887E-02, num_matchable 1.466E+02, num_unmatchable 3.501E+02, confidence 1.544E-01, row_norm 9.829E-01}
[06/15/2024 10:07:54 gluefactory INFO] [Validation] {match_recall 9.743E-01, match_precision 8.621E-01, accuracy 9.753E-01, average_precision 8.391E-01, loss/total 9.035E-02, loss/last 9.035E-02, loss/assignment_nll 9.035E-02, loss/nll_pos 1.226E-01, loss/nll_neg 5.807E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.782E-01}
[06/15/2024 10:07:54 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 10:07:55 gluefactory INFO] New best val: loss/total=0.09035481275945557
[06/15/2024 10:15:23 gluefactory INFO] [Validation] {match_recall 9.742E-01, match_precision 8.615E-01, accuracy 9.752E-01, average_precision 8.386E-01, loss/total 9.056E-02, loss/last 9.056E-02, loss/assignment_nll 9.056E-02, loss/nll_pos 1.226E-01, loss/nll_neg 5.855E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.782E-01}
[06/15/2024 10:15:24 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_32_50984.tar
[06/15/2024 10:15:25 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_25_40000.tar
[06/15/2024 10:15:25 gluefactory INFO] Starting epoch 33
[06/15/2024 10:15:25 gluefactory INFO] lr changed from 5.011872336272722e-06 to 3.981071705534972e-06
[06/15/2024 10:15:30 gluefactory INFO] [E 33 | it 0] loss {total 4.488E-01, last 6.713E-02, assignment_nll 6.713E-02, nll_pos 7.169E-02, nll_neg 6.256E-02, num_matchable 1.228E+02, num_unmatchable 3.732E+02, confidence 1.575E-01, row_norm 9.830E-01}
[06/15/2024 10:18:53 gluefactory INFO] [E 33 | it 100] loss {total 6.525E-01, last 2.792E-01, assignment_nll 2.792E-01, nll_pos 4.688E-01, nll_neg 8.967E-02, num_matchable 1.224E+02, num_unmatchable 3.775E+02, confidence 1.391E-01, row_norm 9.630E-01}
[06/15/2024 10:22:14 gluefactory INFO] [E 33 | it 200] loss {total 4.647E-01, last 5.060E-02, assignment_nll 5.060E-02, nll_pos 5.019E-02, nll_neg 5.100E-02, num_matchable 1.269E+02, num_unmatchable 3.648E+02, confidence 1.619E-01, row_norm 9.788E-01}
[06/15/2024 10:25:36 gluefactory INFO] [E 33 | it 300] loss {total 5.158E-01, last 1.021E-01, assignment_nll 1.021E-01, nll_pos 1.367E-01, nll_neg 6.753E-02, num_matchable 1.114E+02, num_unmatchable 3.879E+02, confidence 1.521E-01, row_norm 9.748E-01}
[06/15/2024 10:28:57 gluefactory INFO] [E 33 | it 400] loss {total 4.960E-01, last 7.879E-02, assignment_nll 7.879E-02, nll_pos 9.683E-02, nll_neg 6.075E-02, num_matchable 1.105E+02, num_unmatchable 3.869E+02, confidence 1.539E-01, row_norm 9.799E-01}
[06/15/2024 10:32:18 gluefactory INFO] [E 33 | it 500] loss {total 4.704E-01, last 6.009E-02, assignment_nll 6.009E-02, nll_pos 6.226E-02, nll_neg 5.792E-02, num_matchable 1.236E+02, num_unmatchable 3.735E+02, confidence 1.605E-01, row_norm 9.825E-01}
[06/15/2024 10:38:12 gluefactory INFO] [Validation] {match_recall 9.741E-01, match_precision 8.640E-01, accuracy 9.757E-01, average_precision 8.409E-01, loss/total 9.010E-02, loss/last 9.010E-02, loss/assignment_nll 9.010E-02, loss/nll_pos 1.231E-01, loss/nll_neg 5.710E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.790E-01}
[06/15/2024 10:38:12 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 10:38:13 gluefactory INFO] New best val: loss/total=0.09009779013544478
[06/15/2024 10:41:35 gluefactory INFO] [E 33 | it 600] loss {total 5.713E-01, last 2.235E-01, assignment_nll 2.235E-01, nll_pos 3.697E-01, nll_neg 7.720E-02, num_matchable 1.194E+02, num_unmatchable 3.757E+02, confidence 1.503E-01, row_norm 9.690E-01}
[06/15/2024 10:44:57 gluefactory INFO] [E 33 | it 700] loss {total 4.607E-01, last 8.956E-02, assignment_nll 8.956E-02, nll_pos 1.264E-01, nll_neg 5.269E-02, num_matchable 1.206E+02, num_unmatchable 3.791E+02, confidence 1.431E-01, row_norm 9.850E-01}
[06/15/2024 10:48:18 gluefactory INFO] [E 33 | it 800] loss {total 4.708E-01, last 7.685E-02, assignment_nll 7.685E-02, nll_pos 9.622E-02, nll_neg 5.748E-02, num_matchable 1.198E+02, num_unmatchable 3.772E+02, confidence 1.563E-01, row_norm 9.793E-01}
[06/15/2024 10:51:39 gluefactory INFO] [E 33 | it 900] loss {total 5.051E-01, last 8.923E-02, assignment_nll 8.923E-02, nll_pos 1.102E-01, nll_neg 6.824E-02, num_matchable 1.240E+02, num_unmatchable 3.722E+02, confidence 1.673E-01, row_norm 9.753E-01}
[06/15/2024 10:55:01 gluefactory INFO] [E 33 | it 1000] loss {total 5.253E-01, last 1.043E-01, assignment_nll 1.043E-01, nll_pos 1.517E-01, nll_neg 5.685E-02, num_matchable 1.297E+02, num_unmatchable 3.676E+02, confidence 1.560E-01, row_norm 9.782E-01}
[06/15/2024 11:00:54 gluefactory INFO] [Validation] {match_recall 9.742E-01, match_precision 8.660E-01, accuracy 9.762E-01, average_precision 8.428E-01, loss/total 8.960E-02, loss/last 8.960E-02, loss/assignment_nll 8.960E-02, loss/nll_pos 1.239E-01, loss/nll_neg 5.532E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.788E-01}
[06/15/2024 11:00:54 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 11:00:54 gluefactory INFO] New best val: loss/total=0.08960223967402457
[06/15/2024 11:04:17 gluefactory INFO] [E 33 | it 1100] loss {total 4.981E-01, last 1.533E-01, assignment_nll 1.533E-01, nll_pos 2.348E-01, nll_neg 7.176E-02, num_matchable 1.337E+02, num_unmatchable 3.654E+02, confidence 1.441E-01, row_norm 9.729E-01}
[06/15/2024 11:07:38 gluefactory INFO] [E 33 | it 1200] loss {total 4.219E-01, last 5.136E-02, assignment_nll 5.136E-02, nll_pos 5.537E-02, nll_neg 4.735E-02, num_matchable 1.258E+02, num_unmatchable 3.703E+02, confidence 1.533E-01, row_norm 9.843E-01}
[06/15/2024 11:11:00 gluefactory INFO] [E 33 | it 1300] loss {total 4.345E-01, last 8.570E-02, assignment_nll 8.570E-02, nll_pos 1.196E-01, nll_neg 5.184E-02, num_matchable 1.312E+02, num_unmatchable 3.666E+02, confidence 1.494E-01, row_norm 9.811E-01}
[06/15/2024 11:14:21 gluefactory INFO] [E 33 | it 1400] loss {total 4.576E-01, last 1.040E-01, assignment_nll 1.040E-01, nll_pos 1.425E-01, nll_neg 6.541E-02, num_matchable 1.434E+02, num_unmatchable 3.552E+02, confidence 1.487E-01, row_norm 9.717E-01}
[06/15/2024 11:17:42 gluefactory INFO] [E 33 | it 1500] loss {total 3.771E-01, last 3.839E-02, assignment_nll 3.839E-02, nll_pos 3.436E-02, nll_neg 4.242E-02, num_matchable 1.520E+02, num_unmatchable 3.449E+02, confidence 1.541E-01, row_norm 9.849E-01}
[06/15/2024 11:23:36 gluefactory INFO] [Validation] {match_recall 9.742E-01, match_precision 8.653E-01, accuracy 9.760E-01, average_precision 8.422E-01, loss/total 8.984E-02, loss/last 8.984E-02, loss/assignment_nll 8.984E-02, loss/nll_pos 1.233E-01, loss/nll_neg 5.634E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.790E-01}
[06/15/2024 11:31:00 gluefactory INFO] [Validation] {match_recall 9.743E-01, match_precision 8.635E-01, accuracy 9.756E-01, average_precision 8.406E-01, loss/total 8.979E-02, loss/last 8.979E-02, loss/assignment_nll 8.979E-02, loss/nll_pos 1.223E-01, loss/nll_neg 5.732E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.788E-01}
[06/15/2024 11:31:02 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_33_52529.tar
[06/15/2024 11:31:02 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_25_40169.tar
[06/15/2024 11:31:02 gluefactory INFO] Starting epoch 34
[06/15/2024 11:31:02 gluefactory INFO] lr changed from 3.981071705534972e-06 to 3.1622776601683788e-06
[06/15/2024 11:31:08 gluefactory INFO] [E 34 | it 0] loss {total 4.981E-01, last 5.982E-02, assignment_nll 5.982E-02, nll_pos 6.559E-02, nll_neg 5.404E-02, num_matchable 1.282E+02, num_unmatchable 3.694E+02, confidence 1.634E-01, row_norm 9.813E-01}
[06/15/2024 11:34:29 gluefactory INFO] [E 34 | it 100] loss {total 5.403E-01, last 1.327E-01, assignment_nll 1.327E-01, nll_pos 1.990E-01, nll_neg 6.639E-02, num_matchable 1.236E+02, num_unmatchable 3.717E+02, confidence 1.634E-01, row_norm 9.756E-01}
[06/15/2024 11:37:51 gluefactory INFO] [E 34 | it 200] loss {total 4.382E-01, last 5.557E-02, assignment_nll 5.557E-02, nll_pos 6.840E-02, nll_neg 4.274E-02, num_matchable 1.328E+02, num_unmatchable 3.622E+02, confidence 1.474E-01, row_norm 9.836E-01}
[06/15/2024 11:41:12 gluefactory INFO] [E 34 | it 300] loss {total 4.046E-01, last 4.867E-02, assignment_nll 4.867E-02, nll_pos 5.260E-02, nll_neg 4.474E-02, num_matchable 1.299E+02, num_unmatchable 3.690E+02, confidence 1.468E-01, row_norm 9.855E-01}
[06/15/2024 11:44:33 gluefactory INFO] [E 34 | it 400] loss {total 4.510E-01, last 7.563E-02, assignment_nll 7.563E-02, nll_pos 9.437E-02, nll_neg 5.689E-02, num_matchable 1.136E+02, num_unmatchable 3.844E+02, confidence 1.475E-01, row_norm 9.824E-01}
[06/15/2024 11:47:55 gluefactory INFO] [E 34 | it 500] loss {total 4.033E-01, last 5.171E-02, assignment_nll 5.171E-02, nll_pos 5.679E-02, nll_neg 4.664E-02, num_matchable 1.288E+02, num_unmatchable 3.712E+02, confidence 1.457E-01, row_norm 9.884E-01}
[06/15/2024 11:53:49 gluefactory INFO] [Validation] {match_recall 9.743E-01, match_precision 8.652E-01, accuracy 9.761E-01, average_precision 8.422E-01, loss/total 8.933E-02, loss/last 8.933E-02, loss/assignment_nll 8.933E-02, loss/nll_pos 1.231E-01, loss/nll_neg 5.560E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.789E-01}
[06/15/2024 11:53:49 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 11:53:49 gluefactory INFO] New best val: loss/total=0.08932770389111636
[06/15/2024 11:57:12 gluefactory INFO] [E 34 | it 600] loss {total 4.657E-01, last 7.782E-02, assignment_nll 7.782E-02, nll_pos 1.027E-01, nll_neg 5.293E-02, num_matchable 1.207E+02, num_unmatchable 3.716E+02, confidence 1.586E-01, row_norm 9.832E-01}
[06/15/2024 12:00:33 gluefactory INFO] [E 34 | it 700] loss {total 4.652E-01, last 6.937E-02, assignment_nll 6.937E-02, nll_pos 8.862E-02, nll_neg 5.011E-02, num_matchable 1.240E+02, num_unmatchable 3.754E+02, confidence 1.453E-01, row_norm 9.840E-01}
[06/15/2024 12:03:55 gluefactory INFO] [E 34 | it 800] loss {total 4.251E-01, last 5.764E-02, assignment_nll 5.764E-02, nll_pos 6.082E-02, nll_neg 5.446E-02, num_matchable 1.242E+02, num_unmatchable 3.731E+02, confidence 1.565E-01, row_norm 9.823E-01}
[06/15/2024 12:07:16 gluefactory INFO] [E 34 | it 900] loss {total 4.442E-01, last 9.601E-02, assignment_nll 9.601E-02, nll_pos 1.323E-01, nll_neg 5.976E-02, num_matchable 1.244E+02, num_unmatchable 3.760E+02, confidence 1.442E-01, row_norm 9.814E-01}
[06/15/2024 12:10:37 gluefactory INFO] [E 34 | it 1000] loss {total 4.268E-01, last 5.693E-02, assignment_nll 5.693E-02, nll_pos 7.000E-02, nll_neg 4.386E-02, num_matchable 1.390E+02, num_unmatchable 3.564E+02, confidence 1.585E-01, row_norm 9.807E-01}
[06/15/2024 12:16:30 gluefactory INFO] [Validation] {match_recall 9.743E-01, match_precision 8.664E-01, accuracy 9.763E-01, average_precision 8.433E-01, loss/total 8.953E-02, loss/last 8.953E-02, loss/assignment_nll 8.953E-02, loss/nll_pos 1.233E-01, loss/nll_neg 5.575E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.793E-01}
[06/15/2024 12:19:52 gluefactory INFO] [E 34 | it 1100] loss {total 4.024E-01, last 4.876E-02, assignment_nll 4.876E-02, nll_pos 5.228E-02, nll_neg 4.525E-02, num_matchable 1.290E+02, num_unmatchable 3.701E+02, confidence 1.519E-01, row_norm 9.856E-01}
[06/15/2024 12:23:14 gluefactory INFO] [E 34 | it 1200] loss {total 4.604E-01, last 6.823E-02, assignment_nll 6.823E-02, nll_pos 8.366E-02, nll_neg 5.280E-02, num_matchable 1.218E+02, num_unmatchable 3.737E+02, confidence 1.530E-01, row_norm 9.811E-01}
[06/15/2024 12:26:35 gluefactory INFO] [E 34 | it 1300] loss {total 4.808E-01, last 9.933E-02, assignment_nll 9.933E-02, nll_pos 1.448E-01, nll_neg 5.383E-02, num_matchable 1.254E+02, num_unmatchable 3.697E+02, confidence 1.620E-01, row_norm 9.766E-01}
[06/15/2024 12:29:57 gluefactory INFO] [E 34 | it 1400] loss {total 4.901E-01, last 1.132E-01, assignment_nll 1.132E-01, nll_pos 1.570E-01, nll_neg 6.936E-02, num_matchable 1.423E+02, num_unmatchable 3.562E+02, confidence 1.500E-01, row_norm 9.731E-01}
[06/15/2024 12:33:18 gluefactory INFO] [E 34 | it 1500] loss {total 4.046E-01, last 4.448E-02, assignment_nll 4.448E-02, nll_pos 5.130E-02, nll_neg 3.766E-02, num_matchable 1.422E+02, num_unmatchable 3.546E+02, confidence 1.509E-01, row_norm 9.839E-01}
[06/15/2024 12:39:12 gluefactory INFO] [Validation] {match_recall 9.743E-01, match_precision 8.666E-01, accuracy 9.764E-01, average_precision 8.435E-01, loss/total 8.879E-02, loss/last 8.879E-02, loss/assignment_nll 8.879E-02, loss/nll_pos 1.221E-01, loss/nll_neg 5.550E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.790E-01}
[06/15/2024 12:39:12 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 12:39:13 gluefactory INFO] New best val: loss/total=0.08879160308282487
[06/15/2024 12:46:39 gluefactory INFO] [Validation] {match_recall 9.744E-01, match_precision 8.651E-01, accuracy 9.760E-01, average_precision 8.421E-01, loss/total 8.882E-02, loss/last 8.882E-02, loss/assignment_nll 8.882E-02, loss/nll_pos 1.208E-01, loss/nll_neg 5.681E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.787E-01}
[06/15/2024 12:46:40 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_34_54074.tar
[06/15/2024 12:46:41 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_26_41714.tar
[06/15/2024 12:46:41 gluefactory INFO] Starting epoch 35
[06/15/2024 12:46:41 gluefactory INFO] lr changed from 3.1622776601683788e-06 to 2.5118864315095797e-06
[06/15/2024 12:46:47 gluefactory INFO] [E 35 | it 0] loss {total 5.037E-01, last 1.165E-01, assignment_nll 1.165E-01, nll_pos 1.708E-01, nll_neg 6.216E-02, num_matchable 1.214E+02, num_unmatchable 3.776E+02, confidence 1.481E-01, row_norm 9.819E-01}
[06/15/2024 12:50:08 gluefactory INFO] [E 35 | it 100] loss {total 5.953E-01, last 1.603E-01, assignment_nll 1.603E-01, nll_pos 2.529E-01, nll_neg 6.767E-02, num_matchable 1.320E+02, num_unmatchable 3.621E+02, confidence 1.634E-01, row_norm 9.696E-01}
[06/15/2024 12:53:30 gluefactory INFO] [E 35 | it 200] loss {total 3.850E-01, last 4.452E-02, assignment_nll 4.452E-02, nll_pos 4.374E-02, nll_neg 4.530E-02, num_matchable 1.289E+02, num_unmatchable 3.670E+02, confidence 1.396E-01, row_norm 9.866E-01}
[06/15/2024 12:56:51 gluefactory INFO] [E 35 | it 300] loss {total 4.397E-01, last 5.741E-02, assignment_nll 5.741E-02, nll_pos 6.068E-02, nll_neg 5.414E-02, num_matchable 1.140E+02, num_unmatchable 3.855E+02, confidence 1.482E-01, row_norm 9.827E-01}
[06/15/2024 13:00:13 gluefactory INFO] [E 35 | it 400] loss {total 5.273E-01, last 8.075E-02, assignment_nll 8.075E-02, nll_pos 1.014E-01, nll_neg 6.009E-02, num_matchable 1.124E+02, num_unmatchable 3.838E+02, confidence 1.591E-01, row_norm 9.746E-01}
[06/15/2024 13:03:34 gluefactory INFO] [E 35 | it 500] loss {total 5.760E-01, last 2.006E-01, assignment_nll 2.006E-01, nll_pos 3.336E-01, nll_neg 6.751E-02, num_matchable 1.275E+02, num_unmatchable 3.713E+02, confidence 1.474E-01, row_norm 9.732E-01}
[06/15/2024 13:09:29 gluefactory INFO] [Validation] {match_recall 9.742E-01, match_precision 8.665E-01, accuracy 9.764E-01, average_precision 8.434E-01, loss/total 8.902E-02, loss/last 8.902E-02, loss/assignment_nll 8.902E-02, loss/nll_pos 1.240E-01, loss/nll_neg 5.406E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.785E-01}
[06/15/2024 13:12:52 gluefactory INFO] [E 35 | it 600] loss {total 4.353E-01, last 6.575E-02, assignment_nll 6.575E-02, nll_pos 7.989E-02, nll_neg 5.161E-02, num_matchable 1.153E+02, num_unmatchable 3.819E+02, confidence 1.465E-01, row_norm 9.820E-01}
[06/15/2024 13:16:14 gluefactory INFO] [E 35 | it 700] loss {total 4.158E-01, last 6.523E-02, assignment_nll 6.523E-02, nll_pos 7.958E-02, nll_neg 5.088E-02, num_matchable 1.197E+02, num_unmatchable 3.799E+02, confidence 1.445E-01, row_norm 9.824E-01}
[06/15/2024 13:19:35 gluefactory INFO] [E 35 | it 800] loss {total 4.792E-01, last 7.861E-02, assignment_nll 7.861E-02, nll_pos 9.984E-02, nll_neg 5.739E-02, num_matchable 1.155E+02, num_unmatchable 3.812E+02, confidence 1.581E-01, row_norm 9.823E-01}
[06/15/2024 13:22:57 gluefactory INFO] [E 35 | it 900] loss {total 5.224E-01, last 1.523E-01, assignment_nll 1.523E-01, nll_pos 2.421E-01, nll_neg 6.247E-02, num_matchable 1.205E+02, num_unmatchable 3.764E+02, confidence 1.588E-01, row_norm 9.757E-01}
[06/15/2024 13:23:47 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_35_55000.tar
[06/15/2024 13:23:48 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_27_43259.tar
[06/15/2024 13:26:19 gluefactory INFO] [E 35 | it 1000] loss {total 4.305E-01, last 6.870E-02, assignment_nll 6.870E-02, nll_pos 8.411E-02, nll_neg 5.330E-02, num_matchable 1.291E+02, num_unmatchable 3.686E+02, confidence 1.466E-01, row_norm 9.788E-01}
[06/15/2024 13:32:14 gluefactory INFO] [Validation] {match_recall 9.743E-01, match_precision 8.663E-01, accuracy 9.763E-01, average_precision 8.432E-01, loss/total 8.868E-02, loss/last 8.868E-02, loss/assignment_nll 8.868E-02, loss/nll_pos 1.213E-01, loss/nll_neg 5.605E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.792E-01}
[06/15/2024 13:32:14 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 13:32:14 gluefactory INFO] New best val: loss/total=0.0886847332323638
[06/15/2024 13:35:37 gluefactory INFO] [E 35 | it 1100] loss {total 4.094E-01, last 4.599E-02, assignment_nll 4.599E-02, nll_pos 4.600E-02, nll_neg 4.597E-02, num_matchable 1.312E+02, num_unmatchable 3.659E+02, confidence 1.554E-01, row_norm 9.870E-01}
[06/15/2024 13:38:58 gluefactory INFO] [E 35 | it 1200] loss {total 4.109E-01, last 5.405E-02, assignment_nll 5.405E-02, nll_pos 5.941E-02, nll_neg 4.869E-02, num_matchable 1.224E+02, num_unmatchable 3.734E+02, confidence 1.484E-01, row_norm 9.844E-01}
[06/15/2024 13:42:20 gluefactory INFO] [E 35 | it 1300] loss {total 4.686E-01, last 7.135E-02, assignment_nll 7.135E-02, nll_pos 8.409E-02, nll_neg 5.860E-02, num_matchable 1.289E+02, num_unmatchable 3.677E+02, confidence 1.606E-01, row_norm 9.790E-01}
[06/15/2024 13:45:41 gluefactory INFO] [E 35 | it 1400] loss {total 4.900E-01, last 9.650E-02, assignment_nll 9.650E-02, nll_pos 1.414E-01, nll_neg 5.157E-02, num_matchable 1.449E+02, num_unmatchable 3.532E+02, confidence 1.506E-01, row_norm 9.827E-01}
[06/15/2024 13:49:02 gluefactory INFO] [E 35 | it 1500] loss {total 4.294E-01, last 6.078E-02, assignment_nll 6.078E-02, nll_pos 7.659E-02, nll_neg 4.497E-02, num_matchable 1.485E+02, num_unmatchable 3.465E+02, confidence 1.584E-01, row_norm 9.799E-01}
[06/15/2024 13:54:57 gluefactory INFO] [Validation] {match_recall 9.743E-01, match_precision 8.659E-01, accuracy 9.762E-01, average_precision 8.429E-01, loss/total 8.884E-02, loss/last 8.884E-02, loss/assignment_nll 8.884E-02, loss/nll_pos 1.227E-01, loss/nll_neg 5.501E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.788E-01}
[06/15/2024 14:02:21 gluefactory INFO] [Validation] {match_recall 9.745E-01, match_precision 8.643E-01, accuracy 9.758E-01, average_precision 8.414E-01, loss/total 8.872E-02, loss/last 8.872E-02, loss/assignment_nll 8.872E-02, loss/nll_pos 1.209E-01, loss/nll_neg 5.657E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.789E-01}
[06/15/2024 14:02:23 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_35_55619.tar
[06/15/2024 14:02:23 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_28_44804.tar
[06/15/2024 14:02:23 gluefactory INFO] Starting epoch 36
[06/15/2024 14:02:23 gluefactory INFO] lr changed from 2.5118864315095797e-06 to 1.995262314968879e-06
[06/15/2024 14:02:29 gluefactory INFO] [E 36 | it 0] loss {total 4.770E-01, last 8.386E-02, assignment_nll 8.386E-02, nll_pos 1.135E-01, nll_neg 5.421E-02, num_matchable 1.300E+02, num_unmatchable 3.687E+02, confidence 1.556E-01, row_norm 9.808E-01}
[06/15/2024 14:05:50 gluefactory INFO] [E 36 | it 100] loss {total 4.425E-01, last 6.918E-02, assignment_nll 6.918E-02, nll_pos 8.654E-02, nll_neg 5.182E-02, num_matchable 1.279E+02, num_unmatchable 3.689E+02, confidence 1.547E-01, row_norm 9.787E-01}
[06/15/2024 14:09:11 gluefactory INFO] [E 36 | it 200] loss {total 4.194E-01, last 5.411E-02, assignment_nll 5.411E-02, nll_pos 5.969E-02, nll_neg 4.854E-02, num_matchable 1.316E+02, num_unmatchable 3.638E+02, confidence 1.472E-01, row_norm 9.851E-01}
[06/15/2024 14:12:33 gluefactory INFO] [E 36 | it 300] loss {total 4.516E-01, last 5.059E-02, assignment_nll 5.059E-02, nll_pos 5.015E-02, nll_neg 5.102E-02, num_matchable 1.229E+02, num_unmatchable 3.743E+02, confidence 1.558E-01, row_norm 9.812E-01}
[06/15/2024 14:15:54 gluefactory INFO] [E 36 | it 400] loss {total 4.831E-01, last 6.159E-02, assignment_nll 6.159E-02, nll_pos 6.816E-02, nll_neg 5.502E-02, num_matchable 1.144E+02, num_unmatchable 3.829E+02, confidence 1.536E-01, row_norm 9.824E-01}
[06/15/2024 14:19:16 gluefactory INFO] [E 36 | it 500] loss {total 4.035E-01, last 3.918E-02, assignment_nll 3.918E-02, nll_pos 3.306E-02, nll_neg 4.529E-02, num_matchable 1.336E+02, num_unmatchable 3.661E+02, confidence 1.568E-01, row_norm 9.871E-01}
[06/15/2024 14:25:10 gluefactory INFO] [Validation] {match_recall 9.744E-01, match_precision 8.665E-01, accuracy 9.764E-01, average_precision 8.435E-01, loss/total 8.848E-02, loss/last 8.848E-02, loss/assignment_nll 8.848E-02, loss/nll_pos 1.218E-01, loss/nll_neg 5.516E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.791E-01}
[06/15/2024 14:25:10 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 14:25:11 gluefactory INFO] New best val: loss/total=0.08847504379312882
[06/15/2024 14:28:34 gluefactory INFO] [E 36 | it 600] loss {total 3.865E-01, last 4.888E-02, assignment_nll 4.888E-02, nll_pos 4.927E-02, nll_neg 4.850E-02, num_matchable 1.218E+02, num_unmatchable 3.743E+02, confidence 1.469E-01, row_norm 9.875E-01}
[06/15/2024 14:31:55 gluefactory INFO] [E 36 | it 700] loss {total 3.941E-01, last 4.961E-02, assignment_nll 4.961E-02, nll_pos 5.496E-02, nll_neg 4.425E-02, num_matchable 1.322E+02, num_unmatchable 3.673E+02, confidence 1.399E-01, row_norm 9.862E-01}
[06/15/2024 14:35:17 gluefactory INFO] [E 36 | it 800] loss {total 6.546E-01, last 2.567E-01, assignment_nll 2.567E-01, nll_pos 4.282E-01, nll_neg 8.518E-02, num_matchable 1.170E+02, num_unmatchable 3.812E+02, confidence 1.487E-01, row_norm 9.608E-01}
[06/15/2024 14:38:38 gluefactory INFO] [E 36 | it 900] loss {total 4.690E-01, last 7.260E-02, assignment_nll 7.260E-02, nll_pos 8.548E-02, nll_neg 5.973E-02, num_matchable 1.257E+02, num_unmatchable 3.720E+02, confidence 1.584E-01, row_norm 9.811E-01}
[06/15/2024 14:42:00 gluefactory INFO] [E 36 | it 1000] loss {total 4.330E-01, last 7.596E-02, assignment_nll 7.596E-02, nll_pos 1.024E-01, nll_neg 4.957E-02, num_matchable 1.416E+02, num_unmatchable 3.541E+02, confidence 1.508E-01, row_norm 9.772E-01}
[06/15/2024 14:47:54 gluefactory INFO] [Validation] {match_recall 9.746E-01, match_precision 8.639E-01, accuracy 9.757E-01, average_precision 8.412E-01, loss/total 8.841E-02, loss/last 8.841E-02, loss/assignment_nll 8.841E-02, loss/nll_pos 1.191E-01, loss/nll_neg 5.773E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.789E-01}
[06/15/2024 14:47:54 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 14:47:55 gluefactory INFO] New best val: loss/total=0.08840805891406177
[06/15/2024 14:51:18 gluefactory INFO] [E 36 | it 1100] loss {total 4.327E-01, last 6.643E-02, assignment_nll 6.643E-02, nll_pos 8.336E-02, nll_neg 4.950E-02, num_matchable 1.284E+02, num_unmatchable 3.684E+02, confidence 1.532E-01, row_norm 9.799E-01}
[06/15/2024 14:54:39 gluefactory INFO] [E 36 | it 1200] loss {total 4.194E-01, last 4.941E-02, assignment_nll 4.941E-02, nll_pos 5.510E-02, nll_neg 4.372E-02, num_matchable 1.346E+02, num_unmatchable 3.619E+02, confidence 1.554E-01, row_norm 9.828E-01}
[06/15/2024 14:58:01 gluefactory INFO] [E 36 | it 1300] loss {total 4.549E-01, last 6.784E-02, assignment_nll 6.784E-02, nll_pos 8.254E-02, nll_neg 5.314E-02, num_matchable 1.261E+02, num_unmatchable 3.711E+02, confidence 1.502E-01, row_norm 9.795E-01}
[06/15/2024 15:01:22 gluefactory INFO] [E 36 | it 1400] loss {total 3.958E-01, last 6.449E-02, assignment_nll 6.449E-02, nll_pos 6.271E-02, nll_neg 6.627E-02, num_matchable 1.374E+02, num_unmatchable 3.614E+02, confidence 1.425E-01, row_norm 9.761E-01}
[06/15/2024 15:04:44 gluefactory INFO] [E 36 | it 1500] loss {total 4.566E-01, last 6.621E-02, assignment_nll 6.621E-02, nll_pos 8.433E-02, nll_neg 4.809E-02, num_matchable 1.429E+02, num_unmatchable 3.507E+02, confidence 1.686E-01, row_norm 9.781E-01}
[06/15/2024 15:10:39 gluefactory INFO] [Validation] {match_recall 9.744E-01, match_precision 8.657E-01, accuracy 9.762E-01, average_precision 8.428E-01, loss/total 8.835E-02, loss/last 8.835E-02, loss/assignment_nll 8.835E-02, loss/nll_pos 1.211E-01, loss/nll_neg 5.558E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.789E-01}
[06/15/2024 15:10:39 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 15:10:39 gluefactory INFO] New best val: loss/total=0.08834805904560603
[06/15/2024 15:18:06 gluefactory INFO] [Validation] {match_recall 9.744E-01, match_precision 8.665E-01, accuracy 9.764E-01, average_precision 8.435E-01, loss/total 8.812E-02, loss/last 8.812E-02, loss/assignment_nll 8.812E-02, loss/nll_pos 1.213E-01, loss/nll_neg 5.494E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.790E-01}
[06/15/2024 15:18:06 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 15:18:07 gluefactory INFO] New best val: loss/total=0.0881173466918712
[06/15/2024 15:18:08 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_36_57164.tar
[06/15/2024 15:18:09 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_29_45000.tar
[06/15/2024 15:18:09 gluefactory INFO] Starting epoch 37
[06/15/2024 15:18:09 gluefactory INFO] lr changed from 1.995262314968879e-06 to 1.584893192461113e-06
[06/15/2024 15:18:15 gluefactory INFO] [E 37 | it 0] loss {total 4.750E-01, last 5.991E-02, assignment_nll 5.991E-02, nll_pos 5.891E-02, nll_neg 6.091E-02, num_matchable 1.200E+02, num_unmatchable 3.785E+02, confidence 1.621E-01, row_norm 9.848E-01}
[06/15/2024 15:21:36 gluefactory INFO] [E 37 | it 100] loss {total 4.440E-01, last 5.955E-02, assignment_nll 5.955E-02, nll_pos 6.537E-02, nll_neg 5.373E-02, num_matchable 1.235E+02, num_unmatchable 3.736E+02, confidence 1.561E-01, row_norm 9.831E-01}
[06/15/2024 15:24:57 gluefactory INFO] [E 37 | it 200] loss {total 3.945E-01, last 4.459E-02, assignment_nll 4.459E-02, nll_pos 4.572E-02, nll_neg 4.347E-02, num_matchable 1.323E+02, num_unmatchable 3.630E+02, confidence 1.476E-01, row_norm 9.854E-01}
[06/15/2024 15:28:19 gluefactory INFO] [E 37 | it 300] loss {total 4.336E-01, last 4.930E-02, assignment_nll 4.930E-02, nll_pos 5.429E-02, nll_neg 4.430E-02, num_matchable 1.237E+02, num_unmatchable 3.758E+02, confidence 1.480E-01, row_norm 9.862E-01}
[06/15/2024 15:31:40 gluefactory INFO] [E 37 | it 400] loss {total 4.731E-01, last 9.797E-02, assignment_nll 9.797E-02, nll_pos 1.390E-01, nll_neg 5.694E-02, num_matchable 1.191E+02, num_unmatchable 3.792E+02, confidence 1.427E-01, row_norm 9.767E-01}
[06/15/2024 15:35:01 gluefactory INFO] [E 37 | it 500] loss {total 4.285E-01, last 5.702E-02, assignment_nll 5.702E-02, nll_pos 6.906E-02, nll_neg 4.498E-02, num_matchable 1.298E+02, num_unmatchable 3.687E+02, confidence 1.527E-01, row_norm 9.857E-01}
[06/15/2024 15:40:56 gluefactory INFO] [Validation] {match_recall 9.743E-01, match_precision 8.666E-01, accuracy 9.764E-01, average_precision 8.436E-01, loss/total 8.820E-02, loss/last 8.820E-02, loss/assignment_nll 8.820E-02, loss/nll_pos 1.214E-01, loss/nll_neg 5.498E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.789E-01}
[06/15/2024 15:44:19 gluefactory INFO] [E 37 | it 600] loss {total 3.992E-01, last 4.804E-02, assignment_nll 4.804E-02, nll_pos 4.897E-02, nll_neg 4.712E-02, num_matchable 1.165E+02, num_unmatchable 3.798E+02, confidence 1.480E-01, row_norm 9.879E-01}
[06/15/2024 15:47:40 gluefactory INFO] [E 37 | it 700] loss {total 4.829E-01, last 1.304E-01, assignment_nll 1.304E-01, nll_pos 2.023E-01, nll_neg 5.849E-02, num_matchable 1.295E+02, num_unmatchable 3.704E+02, confidence 1.473E-01, row_norm 9.763E-01}
[06/15/2024 15:51:01 gluefactory INFO] [E 37 | it 800] loss {total 4.869E-01, last 7.529E-02, assignment_nll 7.529E-02, nll_pos 9.372E-02, nll_neg 5.685E-02, num_matchable 1.222E+02, num_unmatchable 3.725E+02, confidence 1.651E-01, row_norm 9.773E-01}
[06/15/2024 15:54:23 gluefactory INFO] [E 37 | it 900] loss {total 4.348E-01, last 6.415E-02, assignment_nll 6.415E-02, nll_pos 7.730E-02, nll_neg 5.100E-02, num_matchable 1.239E+02, num_unmatchable 3.737E+02, confidence 1.542E-01, row_norm 9.794E-01}
[06/15/2024 15:57:44 gluefactory INFO] [E 37 | it 1000] loss {total 3.975E-01, last 5.925E-02, assignment_nll 5.925E-02, nll_pos 7.478E-02, nll_neg 4.373E-02, num_matchable 1.404E+02, num_unmatchable 3.564E+02, confidence 1.448E-01, row_norm 9.836E-01}
[06/15/2024 16:03:38 gluefactory INFO] [Validation] {match_recall 9.743E-01, match_precision 8.672E-01, accuracy 9.765E-01, average_precision 8.440E-01, loss/total 8.827E-02, loss/last 8.827E-02, loss/assignment_nll 8.827E-02, loss/nll_pos 1.215E-01, loss/nll_neg 5.508E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.788E-01}
[06/15/2024 16:07:00 gluefactory INFO] [E 37 | it 1100] loss {total 3.944E-01, last 4.873E-02, assignment_nll 4.873E-02, nll_pos 5.549E-02, nll_neg 4.196E-02, num_matchable 1.375E+02, num_unmatchable 3.621E+02, confidence 1.470E-01, row_norm 9.863E-01}
[06/15/2024 16:10:22 gluefactory INFO] [E 37 | it 1200] loss {total 4.153E-01, last 5.147E-02, assignment_nll 5.147E-02, nll_pos 5.592E-02, nll_neg 4.703E-02, num_matchable 1.282E+02, num_unmatchable 3.681E+02, confidence 1.515E-01, row_norm 9.847E-01}
[06/15/2024 16:13:43 gluefactory INFO] [E 37 | it 1300] loss {total 4.516E-01, last 6.052E-02, assignment_nll 6.052E-02, nll_pos 7.152E-02, nll_neg 4.953E-02, num_matchable 1.369E+02, num_unmatchable 3.577E+02, confidence 1.629E-01, row_norm 9.776E-01}
[06/15/2024 16:17:04 gluefactory INFO] [E 37 | it 1400] loss {total 4.477E-01, last 8.175E-02, assignment_nll 8.175E-02, nll_pos 1.104E-01, nll_neg 5.314E-02, num_matchable 1.393E+02, num_unmatchable 3.592E+02, confidence 1.476E-01, row_norm 9.840E-01}
[06/15/2024 16:20:26 gluefactory INFO] [E 37 | it 1500] loss {total 4.250E-01, last 5.902E-02, assignment_nll 5.902E-02, nll_pos 7.022E-02, nll_neg 4.782E-02, num_matchable 1.445E+02, num_unmatchable 3.512E+02, confidence 1.570E-01, row_norm 9.818E-01}
[06/15/2024 16:26:20 gluefactory INFO] [Validation] {match_recall 9.746E-01, match_precision 8.673E-01, accuracy 9.765E-01, average_precision 8.442E-01, loss/total 8.767E-02, loss/last 8.767E-02, loss/assignment_nll 8.767E-02, loss/nll_pos 1.203E-01, loss/nll_neg 5.504E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.792E-01}
[06/15/2024 16:26:20 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 16:26:21 gluefactory INFO] New best val: loss/total=0.08767416116962404
[06/15/2024 16:33:46 gluefactory INFO] [Validation] {match_recall 9.745E-01, match_precision 8.664E-01, accuracy 9.764E-01, average_precision 8.434E-01, loss/total 8.771E-02, loss/last 8.771E-02, loss/assignment_nll 8.771E-02, loss/nll_pos 1.204E-01, loss/nll_neg 5.504E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.789E-01}
[06/15/2024 16:33:48 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_37_58709.tar
[06/15/2024 16:33:48 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_29_46349.tar
[06/15/2024 16:33:48 gluefactory INFO] Starting epoch 38
[06/15/2024 16:33:48 gluefactory INFO] lr changed from 1.584893192461113e-06 to 1.2589254117941667e-06
[06/15/2024 16:33:54 gluefactory INFO] [E 38 | it 0] loss {total 5.014E-01, last 8.575E-02, assignment_nll 8.575E-02, nll_pos 1.115E-01, nll_neg 5.998E-02, num_matchable 1.313E+02, num_unmatchable 3.673E+02, confidence 1.571E-01, row_norm 9.816E-01}
[06/15/2024 16:37:15 gluefactory INFO] [E 38 | it 100] loss {total 4.062E-01, last 4.962E-02, assignment_nll 4.962E-02, nll_pos 5.138E-02, nll_neg 4.787E-02, num_matchable 1.258E+02, num_unmatchable 3.730E+02, confidence 1.486E-01, row_norm 9.846E-01}
[06/15/2024 16:40:36 gluefactory INFO] [E 38 | it 200] loss {total 5.498E-01, last 1.682E-01, assignment_nll 1.682E-01, nll_pos 2.726E-01, nll_neg 6.392E-02, num_matchable 1.291E+02, num_unmatchable 3.660E+02, confidence 1.474E-01, row_norm 9.707E-01}
[06/15/2024 16:43:58 gluefactory INFO] [E 38 | it 300] loss {total 4.322E-01, last 7.542E-02, assignment_nll 7.542E-02, nll_pos 9.120E-02, nll_neg 5.965E-02, num_matchable 1.195E+02, num_unmatchable 3.802E+02, confidence 1.415E-01, row_norm 9.809E-01}
[06/15/2024 16:47:19 gluefactory INFO] [E 38 | it 400] loss {total 4.402E-01, last 5.506E-02, assignment_nll 5.506E-02, nll_pos 6.677E-02, nll_neg 4.335E-02, num_matchable 1.240E+02, num_unmatchable 3.735E+02, confidence 1.526E-01, row_norm 9.832E-01}
[06/15/2024 16:50:41 gluefactory INFO] [E 38 | it 500] loss {total 4.098E-01, last 5.220E-02, assignment_nll 5.220E-02, nll_pos 6.038E-02, nll_neg 4.402E-02, num_matchable 1.319E+02, num_unmatchable 3.662E+02, confidence 1.496E-01, row_norm 9.860E-01}
[06/15/2024 16:56:35 gluefactory INFO] [Validation] {match_recall 9.746E-01, match_precision 8.676E-01, accuracy 9.766E-01, average_precision 8.445E-01, loss/total 8.766E-02, loss/last 8.766E-02, loss/assignment_nll 8.766E-02, loss/nll_pos 1.213E-01, loss/nll_neg 5.403E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.791E-01}
[06/15/2024 16:56:36 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 16:56:36 gluefactory INFO] New best val: loss/total=0.0876632743231602
[06/15/2024 16:59:59 gluefactory INFO] [E 38 | it 600] loss {total 4.185E-01, last 6.010E-02, assignment_nll 6.010E-02, nll_pos 7.401E-02, nll_neg 4.620E-02, num_matchable 1.253E+02, num_unmatchable 3.703E+02, confidence 1.547E-01, row_norm 9.837E-01}
[06/15/2024 17:03:20 gluefactory INFO] [E 38 | it 700] loss {total 5.566E-01, last 2.052E-01, assignment_nll 2.052E-01, nll_pos 3.467E-01, nll_neg 6.380E-02, num_matchable 1.286E+02, num_unmatchable 3.719E+02, confidence 1.401E-01, row_norm 9.740E-01}
[06/15/2024 17:06:42 gluefactory INFO] [E 38 | it 800] loss {total 4.769E-01, last 8.450E-02, assignment_nll 8.450E-02, nll_pos 1.138E-01, nll_neg 5.518E-02, num_matchable 1.250E+02, num_unmatchable 3.712E+02, confidence 1.577E-01, row_norm 9.836E-01}
[06/15/2024 17:10:03 gluefactory INFO] [E 38 | it 900] loss {total 4.550E-01, last 6.527E-02, assignment_nll 6.527E-02, nll_pos 7.174E-02, nll_neg 5.880E-02, num_matchable 1.247E+02, num_unmatchable 3.730E+02, confidence 1.550E-01, row_norm 9.795E-01}
[06/15/2024 17:13:24 gluefactory INFO] [E 38 | it 1000] loss {total 4.562E-01, last 6.594E-02, assignment_nll 6.594E-02, nll_pos 8.113E-02, nll_neg 5.076E-02, num_matchable 1.435E+02, num_unmatchable 3.516E+02, confidence 1.576E-01, row_norm 9.830E-01}
[06/15/2024 17:19:19 gluefactory INFO] [Validation] {match_recall 9.744E-01, match_precision 8.676E-01, accuracy 9.766E-01, average_precision 8.445E-01, loss/total 8.763E-02, loss/last 8.763E-02, loss/assignment_nll 8.763E-02, loss/nll_pos 1.212E-01, loss/nll_neg 5.406E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.792E-01}
[06/15/2024 17:19:19 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/15/2024 17:19:20 gluefactory INFO] New best val: loss/total=0.08762909030004062
[06/15/2024 17:22:43 gluefactory INFO] [E 38 | it 1100] loss {total 4.024E-01, last 5.643E-02, assignment_nll 5.643E-02, nll_pos 7.003E-02, nll_neg 4.284E-02, num_matchable 1.266E+02, num_unmatchable 3.749E+02, confidence 1.388E-01, row_norm 9.866E-01}
[06/15/2024 17:26:04 gluefactory INFO] [E 38 | it 1200] loss {total 4.480E-01, last 7.362E-02, assignment_nll 7.362E-02, nll_pos 9.681E-02, nll_neg 5.043E-02, num_matchable 1.301E+02, num_unmatchable 3.654E+02, confidence 1.556E-01, row_norm 9.833E-01}
[06/15/2024 17:29:05 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_38_60000.tar
[06/15/2024 17:29:06 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_30_47894.tar
[06/15/2024 17:29:26 gluefactory INFO] [E 38 | it 1300] loss {total 5.267E-01, last 1.030E-01, assignment_nll 1.030E-01, nll_pos 1.428E-01, nll_neg 6.310E-02, num_matchable 1.218E+02, num_unmatchable 3.727E+02, confidence 1.655E-01, row_norm 9.674E-01}
[06/15/2024 17:32:48 gluefactory INFO] [E 38 | it 1400] loss {total 4.601E-01, last 8.729E-02, assignment_nll 8.729E-02, nll_pos 1.242E-01, nll_neg 5.035E-02, num_matchable 1.331E+02, num_unmatchable 3.647E+02, confidence 1.503E-01, row_norm 9.837E-01}
[06/15/2024 17:36:09 gluefactory INFO] [E 38 | it 1500] loss {total 3.785E-01, last 4.220E-02, assignment_nll 4.220E-02, nll_pos 3.997E-02, nll_neg 4.443E-02, num_matchable 1.433E+02, num_unmatchable 3.530E+02, confidence 1.500E-01, row_norm 9.823E-01}
[06/15/2024 17:42:04 gluefactory INFO] [Validation] {match_recall 9.744E-01, match_precision 8.670E-01, accuracy 9.765E-01, average_precision 8.440E-01, loss/total 8.776E-02, loss/last 8.776E-02, loss/assignment_nll 8.776E-02, loss/nll_pos 1.210E-01, loss/nll_neg 5.454E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.790E-01}
[06/15/2024 17:49:31 gluefactory INFO] [Validation] {match_recall 9.745E-01, match_precision 8.652E-01, accuracy 9.761E-01, average_precision 8.424E-01, loss/total 8.783E-02, loss/last 8.783E-02, loss/assignment_nll 8.783E-02, loss/nll_pos 1.199E-01, loss/nll_neg 5.577E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.787E-01}
[06/15/2024 17:49:33 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_38_60254.tar
[06/15/2024 17:49:33 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_31_49439.tar
[06/15/2024 17:49:33 gluefactory INFO] Starting epoch 39
[06/15/2024 17:49:33 gluefactory INFO] lr changed from 1.2589254117941667e-06 to 9.999999999999995e-07
[06/15/2024 17:49:39 gluefactory INFO] [E 39 | it 0] loss {total 4.556E-01, last 7.209E-02, assignment_nll 7.209E-02, nll_pos 8.721E-02, nll_neg 5.697E-02, num_matchable 1.274E+02, num_unmatchable 3.695E+02, confidence 1.589E-01, row_norm 9.834E-01}
[06/15/2024 17:53:00 gluefactory INFO] [E 39 | it 100] loss {total 4.344E-01, last 6.360E-02, assignment_nll 6.360E-02, nll_pos 7.218E-02, nll_neg 5.501E-02, num_matchable 1.317E+02, num_unmatchable 3.655E+02, confidence 1.583E-01, row_norm 9.820E-01}
[06/15/2024 17:56:22 gluefactory INFO] [E 39 | it 200] loss {total 3.975E-01, last 4.632E-02, assignment_nll 4.632E-02, nll_pos 4.758E-02, nll_neg 4.506E-02, num_matchable 1.329E+02, num_unmatchable 3.629E+02, confidence 1.472E-01, row_norm 9.847E-01}
[06/15/2024 17:59:43 gluefactory INFO] [E 39 | it 300] loss {total 5.610E-01, last 1.282E-01, assignment_nll 1.282E-01, nll_pos 1.924E-01, nll_neg 6.414E-02, num_matchable 1.220E+02, num_unmatchable 3.738E+02, confidence 1.674E-01, row_norm 9.740E-01}
[06/15/2024 18:03:05 gluefactory INFO] [E 39 | it 400] loss {total 4.658E-01, last 7.157E-02, assignment_nll 7.157E-02, nll_pos 8.475E-02, nll_neg 5.839E-02, num_matchable 1.064E+02, num_unmatchable 3.917E+02, confidence 1.476E-01, row_norm 9.826E-01}
[06/15/2024 18:06:26 gluefactory INFO] [E 39 | it 500] loss {total 4.433E-01, last 5.074E-02, assignment_nll 5.074E-02, nll_pos 5.107E-02, nll_neg 5.041E-02, num_matchable 1.293E+02, num_unmatchable 3.702E+02, confidence 1.551E-01, row_norm 9.855E-01}
[06/15/2024 18:12:22 gluefactory INFO] [Validation] {match_recall 9.746E-01, match_precision 8.667E-01, accuracy 9.764E-01, average_precision 8.438E-01, loss/total 8.777E-02, loss/last 8.777E-02, loss/assignment_nll 8.777E-02, loss/nll_pos 1.204E-01, loss/nll_neg 5.511E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.791E-01}
[06/15/2024 18:15:45 gluefactory INFO] [E 39 | it 600] loss {total 4.771E-01, last 7.506E-02, assignment_nll 7.506E-02, nll_pos 9.562E-02, nll_neg 5.449E-02, num_matchable 1.177E+02, num_unmatchable 3.777E+02, confidence 1.567E-01, row_norm 9.843E-01}
[06/15/2024 18:19:06 gluefactory INFO] [E 39 | it 700] loss {total 4.613E-01, last 8.633E-02, assignment_nll 8.633E-02, nll_pos 1.208E-01, nll_neg 5.188E-02, num_matchable 1.151E+02, num_unmatchable 3.838E+02, confidence 1.401E-01, row_norm 9.822E-01}
[06/15/2024 18:22:27 gluefactory INFO] [E 39 | it 800] loss {total 5.741E-01, last 2.162E-01, assignment_nll 2.162E-01, nll_pos 3.538E-01, nll_neg 7.863E-02, num_matchable 1.240E+02, num_unmatchable 3.730E+02, confidence 1.465E-01, row_norm 9.664E-01}
[06/15/2024 18:25:49 gluefactory INFO] [E 39 | it 900] loss {total 5.237E-01, last 1.528E-01, assignment_nll 1.528E-01, nll_pos 2.244E-01, nll_neg 8.120E-02, num_matchable 1.243E+02, num_unmatchable 3.750E+02, confidence 1.507E-01, row_norm 9.638E-01}
[06/15/2024 18:29:10 gluefactory INFO] [E 39 | it 1000] loss {total 5.469E-01, last 2.092E-01, assignment_nll 2.092E-01, nll_pos 3.485E-01, nll_neg 6.996E-02, num_matchable 1.464E+02, num_unmatchable 3.497E+02, confidence 1.498E-01, row_norm 9.668E-01}
[06/15/2024 18:35:05 gluefactory INFO] [Validation] {match_recall 9.745E-01, match_precision 8.671E-01, accuracy 9.765E-01, average_precision 8.442E-01, loss/total 8.770E-02, loss/last 8.770E-02, loss/assignment_nll 8.770E-02, loss/nll_pos 1.204E-01, loss/nll_neg 5.498E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.792E-01}
[06/15/2024 18:38:27 gluefactory INFO] [E 39 | it 1100] loss {total 4.000E-01, last 5.574E-02, assignment_nll 5.574E-02, nll_pos 6.291E-02, nll_neg 4.857E-02, num_matchable 1.331E+02, num_unmatchable 3.668E+02, confidence 1.465E-01, row_norm 9.862E-01}
[06/15/2024 18:41:49 gluefactory INFO] [E 39 | it 1200] loss {total 4.428E-01, last 5.702E-02, assignment_nll 5.702E-02, nll_pos 6.889E-02, nll_neg 4.514E-02, num_matchable 1.255E+02, num_unmatchable 3.705E+02, confidence 1.537E-01, row_norm 9.842E-01}
[06/15/2024 18:45:10 gluefactory INFO] [E 39 | it 1300] loss {total 4.867E-01, last 7.481E-02, assignment_nll 7.481E-02, nll_pos 9.345E-02, nll_neg 5.618E-02, num_matchable 1.262E+02, num_unmatchable 3.686E+02, confidence 1.599E-01, row_norm 9.782E-01}
[06/15/2024 18:48:31 gluefactory INFO] [E 39 | it 1400] loss {total 4.290E-01, last 5.203E-02, assignment_nll 5.203E-02, nll_pos 5.197E-02, nll_neg 5.210E-02, num_matchable 1.353E+02, num_unmatchable 3.624E+02, confidence 1.494E-01, row_norm 9.837E-01}
[06/15/2024 18:51:53 gluefactory INFO] [E 39 | it 1500] loss {total 4.162E-01, last 5.290E-02, assignment_nll 5.290E-02, nll_pos 6.278E-02, nll_neg 4.303E-02, num_matchable 1.432E+02, num_unmatchable 3.530E+02, confidence 1.547E-01, row_norm 9.813E-01}
[06/15/2024 18:57:47 gluefactory INFO] [Validation] {match_recall 9.746E-01, match_precision 8.675E-01, accuracy 9.766E-01, average_precision 8.445E-01, loss/total 8.765E-02, loss/last 8.765E-02, loss/assignment_nll 8.765E-02, loss/nll_pos 1.209E-01, loss/nll_neg 5.443E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.792E-01}
[06/15/2024 19:05:14 gluefactory INFO] [Validation] {match_recall 9.745E-01, match_precision 8.663E-01, accuracy 9.763E-01, average_precision 8.434E-01, loss/total 8.777E-02, loss/last 8.777E-02, loss/assignment_nll 8.777E-02, loss/nll_pos 1.203E-01, loss/nll_neg 5.525E-02, loss/num_matchable 1.258E+02, loss/num_unmatchable 3.714E+02, loss/row_norm 9.791E-01}
[06/15/2024 19:05:16 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_39_61799.tar
[06/15/2024 19:05:16 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_32_50000.tar
[06/15/2024 19:05:16 gluefactory INFO] Finished training on process 0.
"""

base_image_dir = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/"

def parse_log_data(log_data):
    """
    Parses raw log data into a structured pandas DataFrame.
    
    Args:
        log_data (str): Multiline string containing raw log entries.
    
    Returns:
        pd.DataFrame: DataFrame with structured log data.
    """
    # Regular expression pattern to extract essential elements from each log entry
    # pattern = (
    #     r'\[(?P<date>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) gluefactory INFO\] '  # Capture log date and time
    #     r'\[(E (?P<epoch>\d+) \| it (?P<iteration>\d+))? ?(Validation)?\] '     # Capture epoch, iteration, and check for 'Validation'
    #     r'\{(?P<data>.+)\}'                                                      # Capture the data block in curly braces
    # )
    pattern = (
    r'\[(?P<date>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) gluefactory INFO\] '  # Capture log date and time
    r'\[?(E (?P<epoch>\d+) \| it (?P<iteration>\d+))? ?(Validation)?\]? '     # Optionally match epoch, iteration, and 'Validation' within brackets
    r'\{(?P<data>.+)\}'                                                      # Capture the data block in curly braces
    )
    
    rows = []  # List to store parsed data as dictionaries

    # Iterate through each line of the log data
    for line in log_data.splitlines():
        match = re.search(pattern, line)  # Search for pattern in the line
        if match:
            row_data = match.groupdict()  # Extract matched groups as a dictionary
            # print(f"Matched row: {row_data}")  # Debug output

            data_dict = {}
            # Extract key-value pairs from the data part using a regex
            # This captures two possible keys (if present) and the corresponding value formatted as a float in scientific notation
            details = re.findall(r'(\w+)/?(\w+)? (\d+\.\d+E[+-]?\d+)', row_data['data'])
            for key1, key2, value in details:
                # If a second key part exists, concatenate with the first (e.g., 'loss/total'), otherwise use the first key
                if key2:
                    key = f"{key1}/{key2}"
                else:
                    key = key1
                data_dict[key] = float(value.replace('E', 'e'))  # Convert the string value to a float and add to the dictionary

            # print(f"Extracted data dict: {data_dict}")  # Debug output

            # Update the original data row with parsed key-value pairs
            row_data.update(data_dict)
            # Remove the raw data string since it's already parsed into individual elements
            row_data.pop('data')
            # Append the updated row data to the list of rows
            rows.append(row_data)
        else:
            pass
            # print(f"No match for line: {line}")  # Debug for unmatched lines

    # Create a DataFrame from the list of row dictionaries
    df = pd.DataFrame(rows)
    # print(f"DataFrame columns before type conversion: {df.columns}")  # Debug output

    # Convert the 'date' column to datetime objects for better manipulation and filtering
    df['date'] = pd.to_datetime(df['date'])
    # Identify columns that are numeric and convert their data types appropriately for analysis
    numeric_cols = df.columns.difference(['date', 'epoch', 'iteration', 'Validation'])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # print(f"DataFrame after type conversion: {df.head()}")  # Debug output
    
    return df

def plot_loss(df):
    """ Plots the total loss over iterations grouped by epochs. """
    # Check if 'Validation' column exists and filter out validation rows for loss plotting
    # print("\n\n\n", df, df.shape)
    if 'Validation' in df.columns:
        df = df[df['Validation'].isna()]
    
    if 'epoch' in df.columns and 'iteration' in df.columns and 'total' in df.columns:
        for epoch in df['epoch'].unique():
            epoch_data = df[df['epoch'] == epoch]
            plt.plot(epoch_data['iteration'], epoch_data['total'], label=f'Epoch {epoch}')
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.title('Total Loss over Iterations by Epoch')
        plt.legend()
        plt.show()
    else:
        print("Required columns are missing in DataFrame")


def parse_validation_logs(log_data):
    """
    Parses raw log data to extract validation entries and map them to custom 'epoch_iteration' keys.
    
    Args:
        log_data (str): Multiline string containing raw log entries.
        
    Returns:
        pd.DataFrame: DataFrame containing structured log data for validation entries with custom keys.
    """
    pattern = (
        r'\[(?P<date>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) gluefactory INFO\] '  # Capture log date and time
        r'\[Validation\] '  # Identify validation logs
        r'\{(?P<data>.+)\}'  # Capture the data block in curly braces
    )
    
    rows = []
    validation_counter = 0

    # Define the key sequence based on your specified pattern
    epoch_patterns = ["_0", "500", "1000", "final"] # * 39 + ["500", "1000", "final", "final_final"]
    epoch_num = 0
    it_count = 0

    # Iterate through each line of the log data
    for line in log_data.splitlines():
        if "[Validation]" in line:
            # print(f"found a validation line: {line}")
            match = re.search(pattern, line)
            if match:
                # print("found a match!")
                row_data = match.groupdict()

                if it_count % 4 == 0 and it_count != 0:
                    epoch_num += 1
                    it_count /= 4 
                
                # Determine the custom epoch_iteration label
                epoch_iteration = str(epoch_num) + epoch_patterns[it_count]
                
                # Parse data block into key-value pairs
                data_dict = {key: float(value.replace('E', 'e')) for key, value in re.findall(r'(\w+/\w+|\w+) (\d+\.\d+E[+-]?\d+)', row_data['data'])}
                
                # Update row data and assign custom key
                row_data.update(data_dict)
                row_data['epoch_iteration'] = epoch_iteration
                row_data.pop('data', None)  # Remove the original data string
                
                rows.append(row_data)
                validation_counter += 1  # Increment the counter after processing a validation log

    # Create DataFrame
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])  # Convert date string to datetime object
    
    return df

# Function to plot data
def plot_validation_loss(df, exp, dataKey='loss/total'):
    """
    Plots the validation loss from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the log data.
    """
    # print("DataFrame shape:", df.shape)
    
    # Existing line before the changes:
    # print(df[['epoch_iteration', 'loss/total']])
    # for col in df.columns:
    #     print(col)

    ## debugging
    # New check and changes:
    # if 'epoch_iteration' in df.columns and dataKey in df.columns:
    #     print(df[['epoch_iteration', dataKey]])
    #     # proceed with plotting
    # else:
    #     print("Required columns are missing: 'epoch_iteration' or specific data key")


    # Adjusting epoch_iteration labels
    epochs_per_cycle = 4
    iteration_per_epoch = 4

    for i in range(len(df)):
        epoch_num = i // iteration_per_epoch
        if i % iteration_per_epoch == 3:
            iteration_label = 'final'
        else:
            iteration_label = df.loc[i, 'epoch_iteration']
        if i == len(df) - 1:
            df.loc[i, 'epoch_iteration'] = 'final_final'
        else:
            df.loc[i, 'epoch_iteration'] = f"{epoch_num}_{iteration_label}"

    for col in df.columns:
        print(col)
    print(f"looking for {dataKey}")

    if dataKey in df.columns:
        print("Plotting loss total:")
        print("Epoch Iterations:", df['epoch_iteration'].unique())
        print("Loss Total Stats:", df[dataKey].describe())
        
        plt.figure(figsize=(15, 5))
        # plt.plot(df['epoch_iteration'], df['loss/total'], marker='o', linestyle='-', color='b')
        plt.plot(df['epoch_iteration'], df[dataKey], marker='o', linestyle='-', color='b')
        plt.title(f'{dataKey} over Custom Epoch Iterations ')
        plt.xlabel('Epoch_Iteration')
        plt.ylabel(dataKey)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        # plt.title('Validation Loss over Custom Epoch Iterations')
        # plt.xlabel('Epoch_Iteration')
        # plt.ylabel('Loss Total')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        dataKey = dataKey.replace("/", "_")
        # {base_image_dir}
        plt.savefig(f'{exp}_validation_loss_plot_{dataKey}.jpg')  # Saves the plot as a JPEG file
        plt.close()  # Close the plot to free up memory
        print(f"Plot saved as 'validation_loss_plot_{dataKey}.jpg'.")
        # plt.show()
    else:
        print("Loss total data not available in DataFrame.")


def create_multiplot(image_paths, titles, output_filename):
    """
    Creates a multiplot from given image files and saves it.

    Args:
        image_paths (list of str): List of paths to the image files.
        titles (list of str): Titles for each subplot.
        output_filename (str): Path to save the combined image.
    """
    plt.figure(figsize=(20, 10))  # Adjust size as necessary
    num_images = len(image_paths)
    for i in range(num_images):
        img = mpimg.imread(image_paths[i])
        plt.subplot(2, 2, i + 1)  # Adjust grid dimensions as necessary
        plt.imshow(img)
        plt.title(titles[i])
        plt.axis('off')  # Hide axes

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Combined plot saved as {output_filename}")


# df = parse_log_data(log_data)
df = parse_validation_logs(log_data)
exp = "densehomog_"

"""
DF keys:
date
epoch
iteration
match_recall
match_precision
accuracy
average_precision
loss/total
loss/last
loss/assignment_nll
loss/nll_pos
loss/nll_neg
loss/num_matchable
loss/num_unmatchable
loss/row_norm
"""

"""
match_recall 0.000E+00, match_precision 0.000E+00, accuracy 9.540E-01, average_precision 0.000E+00, loss/total 6.332E+00, loss/last 6.332E+00, loss/assignment_nll 6.332E+00, loss/nll_pos 1.210E+01, loss/nll_neg 5.683E-01, loss/num_matchable 7.420E+01, loss/num_unmatchable 1.617E+03, loss/row_norm 6.509E-01}
"""
####################################################################################
### Table summary

# #Print the initial DataFrame
# print("Initial DataFrame:")
# print(df)

# # Check for non-numeric columns
# print("\nColumns and their data types:")
# print(df.dtypes)

# Ensure all relevant columns are numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# # Print the DataFrame after conversion to numeric
# print("\nDataFrame after converting to numeric:")
# print(df)


# # Print the DataFrame after handling NaN values
# print("\nDataFrame after handling NaN values:")
# print(df)

# Calculate statistical summaries
dfsummary = df.describe().transpose()
dfsummary['min'] = df.min()
dfsummary['max'] = df.max()
dfsummary['mean'] = df.mean()
dfsummary['std'] = df.std()

# Select specific metrics
selected_metrics = ['loss/total', 'loss/last', 'loss/assignment_nll', 'loss/nll_pos', 'loss/nll_neg',
                    'loss/num_matchable', 'loss/num_unmatchable', 'loss/row_norm',
                    'match_recall', 'match_precision', 'accuracy', 'average_precision']

dfsummary = dfsummary.loc[selected_metrics]

# Print the summary table
print("\nSummary statistics table:")
print(dfsummary)

# Save the summary table to a LaTeX file
with open('summary_stats.tex', 'w') as f:
    f.write(dfsummary.to_latex())

####################################################################################

### Just the loss
# for col in df.columns:
#     print(col)

# def plot_loss_through_time(df, exp, dataKey):
#     # Assuming 'epoch' is a column in df indicating the epoch number
#     # Make sure 'epoch' is sorted if not already
#     # df = df.sort_values('epoch_iteration')
#     epochs = range(1, len(df['loss/total']) + 1)
    
#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs, df[dataKey], marker='o', linestyle='-')
#     plt.title(f'Validation Loss Through Time for {exp}')
#     plt.xlabel('Epoch')
#     plt.ylabel('Total Loss')
#     plt.grid(True)
    
#     # Save the plot with a dynamic filename
#     filename = f'{base_image_dir}{exp}_validation_loss_plot_{dataKey}.jpg'
#     plt.savefig("loss.jpg")
#     plt.show()

# # Example usage:
# plot_loss_through_time(df, exp, dataKey='loss/total')

### Performance metrics code
# acc_keys = ["accuracy", "match_precision", "match_recall", "average_precision"]
# image_paths = []

# for key in acc_keys:
#     plot_validation_loss(df, exp, key)
#     image_paths.append(f'{base_image_dir}{exp}validation_loss_plot_{key}.jpg')

# # # Paths to your images
# # image_paths = [
# #     f'{base_image_dir}{exp}validation_loss_plot_accuracy.jpg',
# #     f'{base_image_dir}{exp}validation_loss_plot_match_precision.jpg',
# #     f'{base_image_dir}{exp}validation_loss_plot_match_recall.jpg',
# #     f'{base_image_dir}{exp}validation_loss_plot_average_precision.jpg'
# # ]

# # Titles for each subplot
# titles = [
#     'Accuracy', 
#     'Match Precision', 
#     'Match Recall', 
#     'Average Precision'
# ]

# # Filename to save the combined plot
# output_filename = f'{exp}performance_metrics.jpg'

# # Create the multiplot
# create_multiplot(image_paths, titles, output_filename)
###

####################################################################################
#### Plot loss metrics
# make plots on just validaiton
# df = parse_validation_logs(log_data)

# loss_keys = ["loss/total", "loss/last", "loss/assignment_nll", "loss/nll_pos", "loss/nll_neg"]
# loss_image_paths = []

# for key in loss_keys:
#     plot_validation_loss(df, key)
#     dataKey = key.replace("/", "_")
#     loss_image_paths.append(f'/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/{exp}_validation_loss_plot_{dataKey}.jpg')

# plot_loss(df)
# # Paths to the loss metric images
# loss_image_paths = [
#     f'/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/{exp}validation_loss_plot_loss_total.jpg',
#     f'/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/{exp}validation_loss_plot_loss_last.jpg',
#     f'/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/{exp}validation_loss_plot_loss_nll_pos.jpg',
#     f'/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/{exp}validation_loss_plot_loss_nll_neg.jpg'
# # ]

# # Titles for each of the loss metric plots
# loss_titles = [
#     'Total Loss',
#     'Last Loss',
#     'NLL Positive',
#     'NLL Negative'
# ]

# # Filename to save the combined loss metrics plot
# loss_output_filename = f'/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/{exp}loss_metrics.jpg'

# # Call the function to create and save the multiplot
# create_multiplot(loss_image_paths, loss_titles, loss_output_filename)