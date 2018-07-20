<<<<<<< HEAD
Python 3.6.2 |Anaconda custom (x86_64)| (default, Jul 20 2017, 13:14:59) 
[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)] on darwin
Type "copyright", "credits" or "license()" for more information.
>>> 
 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
>>> df.head()
   8  4   5   41789  PsiGS  0.05    5e-05  0.118428867834  0.0415542078994  \
0  8  4   6   61229  PsiGS  0.05  0.00005        0.045801         0.002854   
1  8  4   8  168025  PsiGS  0.05  0.00005       -0.001065         0.000462   
2  8  4   9  169413  PsiGS  0.05  0.00005       -0.000397         0.002028   
3  8  4  10  169413  PsiGS  0.05  0.00005       -0.001708         0.000483   
4  8  4  11  169413  PsiGS  0.05  0.00005       -0.000396         0.002029   

   0.00191224002686  0.020920624443  0.0034918738434  0.000111185335101  
0          0.000112        0.010831         0.001375           0.000006  
1          0.000005        0.004241         0.002757           0.000003  
2          0.000012        0.004154         0.003183           0.000004  
3          0.000005        0.004178         0.002915           0.000003  
4          0.000012        0.004152         0.002937           0.000003  
>>> 
>>> 
>>> 
 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
>>> df.head()
   domainSize  minDepth  maxDepth  numberOfCells  numberOfGridpoints  \
0          15         4         9          10144               90897   
1          15         4        10          10200               91849   
2          15         4        10           6784               61449   
3          15         4        10          16640              147473   
4          15         4        11          17032              151545   

   smoothingN  smoothingEps divideCriterion  divideParameter  energyResidual  \
0           0           1.0             LW1           1000.0        0.000001   
1           0           1.0             LW2           1000.0        0.000001   
2           0           1.0             LW3           1000.0        0.000001   
3           0           1.0             LW1           2000.0        0.000001   
4           0           1.0             LW2           2000.0        0.000001   

   energyErrorGS_analyticPsi  energyErrorGS  psiL2ErrorGS  psiLinfErrorGS  \
0                   0.001347      -0.001179      0.000117    1.392285e-07   
1                   0.001083      -0.001112      0.000076    7.236907e-08   
2                   0.001780      -0.001970      0.000081    1.308703e-07   
3                   0.001095      -0.000928      0.000135    1.423395e-07   
4                   0.000599      -0.000589      0.000068    3.207217e-08   

        GreenReg  
0  volume**(1/3)  
1  volume**(1/3)  
2  volume**(1/3)  
3  volume**(1/3)  
4  volume**(1/3)  
>>> logAversusLogBcolorbyC('energyErrorGS','numberOfGridpoints','divideCriterion')
Traceback (most recent call last):
  File "<pyshell#4>", line 1, in <module>
    logAversusLogBcolorbyC('energyErrorGS','numberOfGridpoints','divideCriterion')
TypeError: logAversusLogBcolorbyC() missing 1 required positional argument: 'C'
>>> logAversusLogBcolorbyC(df,'energyErrorGS','numberOfGridpoints','divideCriterion')

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 122
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 123
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 122
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 123
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
>>> logAversusLogBcolorbyC(df,'energyErrorGS','numberOfGridpoints','divideCriterion',trendline=True)

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 127
    group['trendline'] = p(group['logB'])
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
>>> logAversusLogBcolorbyC(df,'energyErrorGS','numberOfGridpoints','divideCriterion')
>>> logAversusLogBcolorbyC(df.loc[df['GreenReg'] == 'volume**(1/3)'],'energyErrorGS','numberOfGridpoints','divideCriterion')
>>> df.tail()
    domainSize  minDepth  maxDepth  numberOfCells  numberOfGridpoints  \
30          15         4        11          10088               90929   
31          15         4        11          28624              249813   
32          15         4        12          30416              266189   
33          15         4        12          16864              150165   
34          15         4        11          53992              464505   

    smoothingN  smoothingEps divideCriterion  divideParameter  energyResidual  \
30           0         0.005             LW3           2000.0        0.000001   
31           0         0.005             LW1           4000.0        0.000001   
32           0         0.005             LW2           4000.0        0.000001   
33           0         0.005             LW3           4000.0        0.000001   
34           0         0.005             LW1           8000.0        0.000001   

    energyErrorGS_analyticPsi  energyErrorGS  psiL2ErrorGS  psiLinfErrorGS  \
30                  -0.001036       0.350701      1.971000        0.021454   
31                  -0.000771       0.350701      1.971208        0.021459   
32                  -0.000473       0.376939      1.991092        0.017714   
33                  -0.000616       0.369770      1.991150        0.040996   
34                  -0.000491       0.361187      1.978428        0.013732   

      GreenReg  
30  fixed0p005  
31  fixed0p005  
32  fixed0p005  
33  fixed0p005  
34  fixed0p005  
>>> logAversusBcolorbyC(df.loc[df['GreenReg'] == 'volume**(1/3)'],'energyErrorGS','numberOfGridpoints','divideCriterion')

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 107
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
Traceback (most recent call last):
  File "<pyshell#10>", line 1, in <module>
    logAversusBcolorbyC(df.loc[df['GreenReg'] == 'volume**(1/3)'],'energyErrorGS','numberOfGridpoints','divideCriterion')
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 108, in logAversusBcolorbyC
    group.plot(x=B, y='logA', style='o', ax=ax, label='%s = %.2f'%(C,name))
TypeError: must be real number, not str
>>> logAversusLogBcolorbyC(df.loc[df['GreenReg'] == 'volume**(1/3)'],'energyErrorGS','numberOfGridpoints','divideCriterion')
>>> 
>>> 
>>> 
>>> 
>>> df
    domainSize  minDepth  maxDepth  numberOfCells  numberOfGridpoints  \
0           15         4         9          10144               90897   
1           15         4        10          10200               91849   
2           15         4        10           6784               61449   
3           15         4        10          16640              147473   
4           15         4        11          17032              151545   
5           15         4        11          10088               90929   
6           15         4        11          28624              249813   
7           15         4        12          30416              266189   
8           15         4        12          16864              150165   
9           15         4        11          53992              464505   
10          15         4        13          56288              483893   
11          15         4        13          29800              261429   
12          15         4        10          10200               91849   
13          15         4         9          10144               90897   
14          15         4        10          10200               91849   
15          15         4        10           6784               61449   
16          15         4        10          16640              147473   
17          15         4        11          17032              151545   
18          15         4        11          10088               90929   
19          15         4        11          28624              249813   
20          15         4        12          30416              266189   
21          15         4        12          16864              150165   
22          15         4        11          53992              464505   
23          15         4        13          56288              483893   
24          15         4        13          29800              261429   
25          15         4         9          10144               90897   
26          15         4        10          10200               91849   
27          15         4        10           6784               61449   
28          15         4        10          16640              147473   
29          15         4        11          17032              151545   
30          15         4        11          10088               90929   
31          15         4        11          28624              249813   
32          15         4        12          30416              266189   
33          15         4        12          16864              150165   
34          15         4        11          53992              464505   

    smoothingN  smoothingEps divideCriterion  divideParameter  energyResidual  \
0            0         1.000             LW1           1000.0        0.000001   
1            0         1.000             LW2           1000.0        0.000001   
2            0         1.000             LW3           1000.0        0.000001   
3            0         1.000             LW1           2000.0        0.000001   
4            0         1.000             LW2           2000.0        0.000001   
5            0         1.000             LW3           2000.0        0.000001   
6            0         1.000             LW1           4000.0        0.000001   
7            0         1.000             LW2           4000.0        0.000001   
8            0         1.000             LW3           4000.0        0.000001   
9            0         1.000             LW1           8000.0        0.000001   
10           0         1.000             LW2           8000.0        0.000001   
11           0         1.000             LW3           8000.0        0.000001   
12           0         1.000             LW2           1000.0        0.000001   
13           0         1.000             LW1           1000.0        0.000001   
14           0         1.000             LW2           1000.0        0.000001   
15           0         1.000             LW3           1000.0        0.000001   
16           0         1.000             LW1           2000.0        0.000001   
17           0         1.000             LW2           2000.0        0.000001   
18           0         1.000             LW3           2000.0        0.000001   
19           0         1.000             LW1           4000.0        0.000001   
20           0         1.000             LW2           4000.0        0.000001   
21           0         1.000             LW3           4000.0        0.000001   
22           0         1.000             LW1           8000.0        0.000001   
23           0         1.000             LW2           8000.0        0.000001   
24           0         1.000             LW3           8000.0        0.000001   
25           0         0.005             LW1           1000.0        0.000001   
26           0         0.005             LW2           1000.0        0.000001   
27           0         0.005             LW3           1000.0        0.000001   
28           0         0.005             LW1           2000.0        0.000001   
29           0         0.005             LW2           2000.0        0.000001   
30           0         0.005             LW3           2000.0        0.000001   
31           0         0.005             LW1           4000.0        0.000001   
32           0         0.005             LW2           4000.0        0.000001   
33           0         0.005             LW3           4000.0        0.000001   
34           0         0.005             LW1           8000.0        0.000001   

    energyErrorGS_analyticPsi  energyErrorGS  psiL2ErrorGS  psiLinfErrorGS  \
0                    0.001347      -0.001179      0.000117    1.392285e-07   
1                    0.001083      -0.001112      0.000076    7.236907e-08   
2                    0.001780      -0.001970      0.000081    1.308703e-07   
3                    0.001095      -0.000928      0.000135    1.423395e-07   
4                    0.000599      -0.000589      0.000068    3.207217e-08   
5                    0.001036      -0.001099      0.000118    9.634610e-08   
6                    0.000771      -0.000650      0.000114    4.946041e-08   
7                    0.000473      -0.000421      0.000070    1.932295e-08   
8                    0.000616      -0.000630      0.000073    3.414204e-08   
9                    0.000491      -0.000405      0.000076    2.076107e-08   
10                   0.000271      -0.000228      0.000040    5.558403e-09   
11                   1.998420      -0.000406      0.000072    2.525128e-08   
12                   1.995974       0.061347      0.091137    7.849253e-05   
13                  -0.001347       0.061008      0.092482    1.103454e-04   
14                  -0.001083       0.061347      0.091137    7.849253e-05   
15                  -0.001780       0.063191      0.097192    1.629038e-04   
16                  -0.001095       0.060338      0.089158    4.651185e-05   
17                  -0.000599       0.061234      0.089652    4.683014e-05   
18                  -0.001036       0.062180      0.091911    7.962988e-05   
19                  -0.000771       0.060581      0.088711    2.342316e-05   
20                  -0.000473       0.061309      0.089290    2.358815e-05   
21                  -0.000616       0.061484      0.089988    4.695751e-05   
22                  -0.000491       0.060831      0.088570    9.898562e-06   
23                  -0.000271       0.061481      0.089292    9.750734e-06   
24                  -0.000445       0.061526      0.089628    2.616577e-05   
25                  -0.001347       0.319737      1.916042    1.920165e-02   
26                  -0.001083       0.337366      1.966052    4.035160e-02   
27                  -0.001780       0.319737      1.915596    1.919710e-02   
28                  -0.001095       0.337366      1.966145    4.035328e-02   
29                  -0.000599       0.361187      1.978322    1.372939e-02   
30                  -0.001036       0.350701      1.971000    2.145446e-02   
31                  -0.000771       0.350701      1.971208    2.145937e-02   
32                  -0.000473       0.376939      1.991092    1.771377e-02   
33                  -0.000616       0.369770      1.991150    4.099574e-02   
34                  -0.000491       0.361187      1.978428    1.373164e-02   

         GreenReg  
0   volume**(1/3)  
1   volume**(1/3)  
2   volume**(1/3)  
3   volume**(1/3)  
4   volume**(1/3)  
5   volume**(1/3)  
6   volume**(1/3)  
7   volume**(1/3)  
8   volume**(1/3)  
9   volume**(1/3)  
10  volume**(1/3)  
11  volume**(1/3)  
12       fixed0p5  
13       fixed0p5  
14       fixed0p5  
15       fixed0p5  
16       fixed0p5  
17       fixed0p5  
18       fixed0p5  
19       fixed0p5  
20       fixed0p5  
21       fixed0p5  
22       fixed0p5  
23       fixed0p5  
24       fixed0p5  
25     fixed0p005  
26     fixed0p005  
27     fixed0p005  
28     fixed0p005  
29     fixed0p005  
30     fixed0p005  
31     fixed0p005  
32     fixed0p005  
33     fixed0p005  
34     fixed0p005  
>>> logAversusLogBcolorbyC(df.loc[df['GreenReg'] == 'volume**(1/3)'],'energyErrorGS','numberOfGridpoints','divideCriterion')

 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
>>> df
    domainSize  minDepth  maxDepth  numberOfCells  numberOfGridpoints  \
0           15         4         9          10144               90897   
1           15         4        10          10200               91849   
2           15         4        10           6784               61449   
3           15         4        10          16640              147473   
4           15         4        11          17032              151545   
5           15         4        11          10088               90929   
6           15         4        11          28624              249813   
7           15         4        12          30416              266189   
8           15         4        12          16864              150165   
9           15         4        11          53992              464505   
10          15         4        13          56288              483893   
11          15         4        13          29800              261429   
12          15         4        10          10200               91849   
13          15         4         9          10144               90897   
14          15         4        10          10200               91849   
15          15         4        10           6784               61449   
16          15         4        10          16640              147473   
17          15         4        11          17032              151545   
18          15         4        11          10088               90929   
19          15         4        11          28624              249813   
20          15         4        12          30416              266189   
21          15         4        12          16864              150165   
22          15         4        11          53992              464505   
23          15         4        13          56288              483893   
24          15         4        13          29800              261429   
25          15         4         9          10144               90897   
26          15         4        10          10200               91849   
27          15         4        10           6784               61449   
28          15         4        10          16640              147473   
29          15         4        11          17032              151545   
30          15         4        11          10088               90929   
31          15         4        11          28624              249813   
32          15         4        12          30416              266189   
33          15         4        12          16864              150165   
34          15         4        11          53992              464505   

    smoothingN  smoothingEps divideCriterion  divideParameter  energyResidual  \
0            0         1.000             LW1           1000.0        0.000001   
1            0         1.000             LW2           1000.0        0.000001   
2            0         1.000             LW3           1000.0        0.000001   
3            0         1.000             LW1           2000.0        0.000001   
4            0         1.000             LW2           2000.0        0.000001   
5            0         1.000             LW3           2000.0        0.000001   
6            0         1.000             LW1           4000.0        0.000001   
7            0         1.000             LW2           4000.0        0.000001   
8            0         1.000             LW3           4000.0        0.000001   
9            0         1.000             LW1           8000.0        0.000001   
10           0         1.000             LW2           8000.0        0.000001   
11           0         1.000             LW3           8000.0        0.000001   
12           0         1.000             LW2           1000.0        0.000001   
13           0         1.000             LW1           1000.0        0.000001   
14           0         1.000             LW2           1000.0        0.000001   
15           0         1.000             LW3           1000.0        0.000001   
16           0         1.000             LW1           2000.0        0.000001   
17           0         1.000             LW2           2000.0        0.000001   
18           0         1.000             LW3           2000.0        0.000001   
19           0         1.000             LW1           4000.0        0.000001   
20           0         1.000             LW2           4000.0        0.000001   
21           0         1.000             LW3           4000.0        0.000001   
22           0         1.000             LW1           8000.0        0.000001   
23           0         1.000             LW2           8000.0        0.000001   
24           0         1.000             LW3           8000.0        0.000001   
25           0         0.005             LW1           1000.0        0.000001   
26           0         0.005             LW2           1000.0        0.000001   
27           0         0.005             LW3           1000.0        0.000001   
28           0         0.005             LW1           2000.0        0.000001   
29           0         0.005             LW2           2000.0        0.000001   
30           0         0.005             LW3           2000.0        0.000001   
31           0         0.005             LW1           4000.0        0.000001   
32           0         0.005             LW2           4000.0        0.000001   
33           0         0.005             LW3           4000.0        0.000001   
34           0         0.005             LW1           8000.0        0.000001   

    energyErrorGS_analyticPsi  energyErrorGS  psiL2ErrorGS  psiLinfErrorGS  \
0                    0.001347      -0.001179      0.000117    1.392285e-07   
1                    0.001083      -0.001112      0.000076    7.236907e-08   
2                    0.001780      -0.001970      0.000081    1.308703e-07   
3                    0.001095      -0.000928      0.000135    1.423395e-07   
4                    0.000599      -0.000589      0.000068    3.207217e-08   
5                    0.001036      -0.001099      0.000118    9.634610e-08   
6                    0.000771      -0.000650      0.000114    4.946041e-08   
7                    0.000473      -0.000421      0.000070    1.932295e-08   
8                    0.000616      -0.000630      0.000073    3.414204e-08   
9                    0.000491      -0.000405      0.000076    2.076107e-08   
10                   0.000271      -0.000228      0.000040    5.558403e-09   
11                   1.998420      -0.000406      0.000072    2.525128e-08   
12                   1.995974       0.061347      0.091137    7.849253e-05   
13                  -0.001347       0.061008      0.092482    1.103454e-04   
14                  -0.001083       0.061347      0.091137    7.849253e-05   
15                  -0.001780       0.063191      0.097192    1.629038e-04   
16                  -0.001095       0.060338      0.089158    4.651185e-05   
17                  -0.000599       0.061234      0.089652    4.683014e-05   
18                  -0.001036       0.062180      0.091911    7.962988e-05   
19                  -0.000771       0.060581      0.088711    2.342316e-05   
20                  -0.000473       0.061309      0.089290    2.358815e-05   
21                  -0.000616       0.061484      0.089988    4.695751e-05   
22                  -0.000491       0.060831      0.088570    9.898562e-06   
23                  -0.000271       0.061481      0.089292    9.750734e-06   
24                  -0.000445       0.061526      0.089628    2.616577e-05   
25                  -0.001347       0.319737      1.916042    1.920165e-02   
26                  -0.001083       0.337366      1.966052    4.035160e-02   
27                  -0.001780       0.319737      1.915596    1.919710e-02   
28                  -0.001095       0.337366      1.966145    4.035328e-02   
29                  -0.000599       0.361187      1.978322    1.372939e-02   
30                  -0.001036       0.350701      1.971000    2.145446e-02   
31                  -0.000771       0.350701      1.971208    2.145937e-02   
32                  -0.000473       0.376939      1.991092    1.771377e-02   
33                  -0.000616       0.369770      1.991150    4.099574e-02   
34                  -0.000491       0.361187      1.978428    1.373164e-02   

         GreenReg  
0   volume**(1/3)  
1   volume**(1/3)  
2   volume**(1/3)  
3   volume**(1/3)  
4   volume**(1/3)  
5   volume**(1/3)  
6   volume**(1/3)  
7   volume**(1/3)  
8   volume**(1/3)  
9   volume**(1/3)  
10  volume**(1/3)  
11  volume**(1/3)  
12       fixed0p5  
13       fixed0p5  
14       fixed0p5  
15       fixed0p5  
16       fixed0p5  
17       fixed0p5  
18       fixed0p5  
19       fixed0p5  
20       fixed0p5  
21       fixed0p5  
22       fixed0p5  
23       fixed0p5  
24       fixed0p5  
25     fixed0p005  
26     fixed0p005  
27     fixed0p005  
28     fixed0p005  
29     fixed0p005  
30     fixed0p005  
31     fixed0p005  
32     fixed0p005  
33     fixed0p005  
34     fixed0p005  
>>> logAandBversusLogCcolorbyD(df,'energyErrorGS','energyErrorGS_analyticPsi','numberOfGridpoints','divideCriterion')

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 148
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 149
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 148
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 149
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
>>> 
 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
>>> logAandBversusLogCcolorbyD(df,'energyErrorGS','energyErrorGS_analyticPsi','numberOfGridpoints','divideCriterion')

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 148
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 149
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 148
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 149
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
>>> 
 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
>>> logAandBversusLogCcolorbyD(df,'energyErrorGS','energyErrorGS_analyticPsi','numberOfGridpoints','divideCriterion')

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 151
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 152
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 151
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 152
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
>>> 
 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
=======
Python 3.6.2 |Anaconda custom (x86_64)| (default, Jul 20 2017, 13:14:59) 
[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)] on darwin
Type "copyright", "credits" or "license()" for more information.
>>> 
 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
>>> df.head()
   8  4   5   41789  PsiGS  0.05    5e-05  0.118428867834  0.0415542078994  \
0  8  4   6   61229  PsiGS  0.05  0.00005        0.045801         0.002854   
1  8  4   8  168025  PsiGS  0.05  0.00005       -0.001065         0.000462   
2  8  4   9  169413  PsiGS  0.05  0.00005       -0.000397         0.002028   
3  8  4  10  169413  PsiGS  0.05  0.00005       -0.001708         0.000483   
4  8  4  11  169413  PsiGS  0.05  0.00005       -0.000396         0.002029   

   0.00191224002686  0.020920624443  0.0034918738434  0.000111185335101  
0          0.000112        0.010831         0.001375           0.000006  
1          0.000005        0.004241         0.002757           0.000003  
2          0.000012        0.004154         0.003183           0.000004  
3          0.000005        0.004178         0.002915           0.000003  
4          0.000012        0.004152         0.002937           0.000003  
>>> 
>>> 
>>> 
 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
>>> df.head()
   domainSize  minDepth  maxDepth  numberOfCells  numberOfGridpoints  \
0          15         4         9          10144               90897   
1          15         4        10          10200               91849   
2          15         4        10           6784               61449   
3          15         4        10          16640              147473   
4          15         4        11          17032              151545   

   smoothingN  smoothingEps divideCriterion  divideParameter  energyResidual  \
0           0           1.0             LW1           1000.0        0.000001   
1           0           1.0             LW2           1000.0        0.000001   
2           0           1.0             LW3           1000.0        0.000001   
3           0           1.0             LW1           2000.0        0.000001   
4           0           1.0             LW2           2000.0        0.000001   

   energyErrorGS_analyticPsi  energyErrorGS  psiL2ErrorGS  psiLinfErrorGS  \
0                   0.001347      -0.001179      0.000117    1.392285e-07   
1                   0.001083      -0.001112      0.000076    7.236907e-08   
2                   0.001780      -0.001970      0.000081    1.308703e-07   
3                   0.001095      -0.000928      0.000135    1.423395e-07   
4                   0.000599      -0.000589      0.000068    3.207217e-08   

        GreenReg  
0  volume**(1/3)  
1  volume**(1/3)  
2  volume**(1/3)  
3  volume**(1/3)  
4  volume**(1/3)  
>>> logAversusLogBcolorbyC('energyErrorGS','numberOfGridpoints','divideCriterion')
Traceback (most recent call last):
  File "<pyshell#4>", line 1, in <module>
    logAversusLogBcolorbyC('energyErrorGS','numberOfGridpoints','divideCriterion')
TypeError: logAversusLogBcolorbyC() missing 1 required positional argument: 'C'
>>> logAversusLogBcolorbyC(df,'energyErrorGS','numberOfGridpoints','divideCriterion')

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 122
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 123
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 122
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 123
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
>>> logAversusLogBcolorbyC(df,'energyErrorGS','numberOfGridpoints','divideCriterion',trendline=True)

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 127
    group['trendline'] = p(group['logB'])
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
>>> logAversusLogBcolorbyC(df,'energyErrorGS','numberOfGridpoints','divideCriterion')
>>> logAversusLogBcolorbyC(df.loc[df['GreenReg'] == 'volume**(1/3)'],'energyErrorGS','numberOfGridpoints','divideCriterion')
>>> df.tail()
    domainSize  minDepth  maxDepth  numberOfCells  numberOfGridpoints  \
30          15         4        11          10088               90929   
31          15         4        11          28624              249813   
32          15         4        12          30416              266189   
33          15         4        12          16864              150165   
34          15         4        11          53992              464505   

    smoothingN  smoothingEps divideCriterion  divideParameter  energyResidual  \
30           0         0.005             LW3           2000.0        0.000001   
31           0         0.005             LW1           4000.0        0.000001   
32           0         0.005             LW2           4000.0        0.000001   
33           0         0.005             LW3           4000.0        0.000001   
34           0         0.005             LW1           8000.0        0.000001   

    energyErrorGS_analyticPsi  energyErrorGS  psiL2ErrorGS  psiLinfErrorGS  \
30                  -0.001036       0.350701      1.971000        0.021454   
31                  -0.000771       0.350701      1.971208        0.021459   
32                  -0.000473       0.376939      1.991092        0.017714   
33                  -0.000616       0.369770      1.991150        0.040996   
34                  -0.000491       0.361187      1.978428        0.013732   

      GreenReg  
30  fixed0p005  
31  fixed0p005  
32  fixed0p005  
33  fixed0p005  
34  fixed0p005  
>>> logAversusBcolorbyC(df.loc[df['GreenReg'] == 'volume**(1/3)'],'energyErrorGS','numberOfGridpoints','divideCriterion')

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 107
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
Traceback (most recent call last):
  File "<pyshell#10>", line 1, in <module>
    logAversusBcolorbyC(df.loc[df['GreenReg'] == 'volume**(1/3)'],'energyErrorGS','numberOfGridpoints','divideCriterion')
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 108, in logAversusBcolorbyC
    group.plot(x=B, y='logA', style='o', ax=ax, label='%s = %.2f'%(C,name))
TypeError: must be real number, not str
>>> logAversusLogBcolorbyC(df.loc[df['GreenReg'] == 'volume**(1/3)'],'energyErrorGS','numberOfGridpoints','divideCriterion')
>>> 
>>> 
>>> 
>>> 
>>> df
    domainSize  minDepth  maxDepth  numberOfCells  numberOfGridpoints  \
0           15         4         9          10144               90897   
1           15         4        10          10200               91849   
2           15         4        10           6784               61449   
3           15         4        10          16640              147473   
4           15         4        11          17032              151545   
5           15         4        11          10088               90929   
6           15         4        11          28624              249813   
7           15         4        12          30416              266189   
8           15         4        12          16864              150165   
9           15         4        11          53992              464505   
10          15         4        13          56288              483893   
11          15         4        13          29800              261429   
12          15         4        10          10200               91849   
13          15         4         9          10144               90897   
14          15         4        10          10200               91849   
15          15         4        10           6784               61449   
16          15         4        10          16640              147473   
17          15         4        11          17032              151545   
18          15         4        11          10088               90929   
19          15         4        11          28624              249813   
20          15         4        12          30416              266189   
21          15         4        12          16864              150165   
22          15         4        11          53992              464505   
23          15         4        13          56288              483893   
24          15         4        13          29800              261429   
25          15         4         9          10144               90897   
26          15         4        10          10200               91849   
27          15         4        10           6784               61449   
28          15         4        10          16640              147473   
29          15         4        11          17032              151545   
30          15         4        11          10088               90929   
31          15         4        11          28624              249813   
32          15         4        12          30416              266189   
33          15         4        12          16864              150165   
34          15         4        11          53992              464505   

    smoothingN  smoothingEps divideCriterion  divideParameter  energyResidual  \
0            0         1.000             LW1           1000.0        0.000001   
1            0         1.000             LW2           1000.0        0.000001   
2            0         1.000             LW3           1000.0        0.000001   
3            0         1.000             LW1           2000.0        0.000001   
4            0         1.000             LW2           2000.0        0.000001   
5            0         1.000             LW3           2000.0        0.000001   
6            0         1.000             LW1           4000.0        0.000001   
7            0         1.000             LW2           4000.0        0.000001   
8            0         1.000             LW3           4000.0        0.000001   
9            0         1.000             LW1           8000.0        0.000001   
10           0         1.000             LW2           8000.0        0.000001   
11           0         1.000             LW3           8000.0        0.000001   
12           0         1.000             LW2           1000.0        0.000001   
13           0         1.000             LW1           1000.0        0.000001   
14           0         1.000             LW2           1000.0        0.000001   
15           0         1.000             LW3           1000.0        0.000001   
16           0         1.000             LW1           2000.0        0.000001   
17           0         1.000             LW2           2000.0        0.000001   
18           0         1.000             LW3           2000.0        0.000001   
19           0         1.000             LW1           4000.0        0.000001   
20           0         1.000             LW2           4000.0        0.000001   
21           0         1.000             LW3           4000.0        0.000001   
22           0         1.000             LW1           8000.0        0.000001   
23           0         1.000             LW2           8000.0        0.000001   
24           0         1.000             LW3           8000.0        0.000001   
25           0         0.005             LW1           1000.0        0.000001   
26           0         0.005             LW2           1000.0        0.000001   
27           0         0.005             LW3           1000.0        0.000001   
28           0         0.005             LW1           2000.0        0.000001   
29           0         0.005             LW2           2000.0        0.000001   
30           0         0.005             LW3           2000.0        0.000001   
31           0         0.005             LW1           4000.0        0.000001   
32           0         0.005             LW2           4000.0        0.000001   
33           0         0.005             LW3           4000.0        0.000001   
34           0         0.005             LW1           8000.0        0.000001   

    energyErrorGS_analyticPsi  energyErrorGS  psiL2ErrorGS  psiLinfErrorGS  \
0                    0.001347      -0.001179      0.000117    1.392285e-07   
1                    0.001083      -0.001112      0.000076    7.236907e-08   
2                    0.001780      -0.001970      0.000081    1.308703e-07   
3                    0.001095      -0.000928      0.000135    1.423395e-07   
4                    0.000599      -0.000589      0.000068    3.207217e-08   
5                    0.001036      -0.001099      0.000118    9.634610e-08   
6                    0.000771      -0.000650      0.000114    4.946041e-08   
7                    0.000473      -0.000421      0.000070    1.932295e-08   
8                    0.000616      -0.000630      0.000073    3.414204e-08   
9                    0.000491      -0.000405      0.000076    2.076107e-08   
10                   0.000271      -0.000228      0.000040    5.558403e-09   
11                   1.998420      -0.000406      0.000072    2.525128e-08   
12                   1.995974       0.061347      0.091137    7.849253e-05   
13                  -0.001347       0.061008      0.092482    1.103454e-04   
14                  -0.001083       0.061347      0.091137    7.849253e-05   
15                  -0.001780       0.063191      0.097192    1.629038e-04   
16                  -0.001095       0.060338      0.089158    4.651185e-05   
17                  -0.000599       0.061234      0.089652    4.683014e-05   
18                  -0.001036       0.062180      0.091911    7.962988e-05   
19                  -0.000771       0.060581      0.088711    2.342316e-05   
20                  -0.000473       0.061309      0.089290    2.358815e-05   
21                  -0.000616       0.061484      0.089988    4.695751e-05   
22                  -0.000491       0.060831      0.088570    9.898562e-06   
23                  -0.000271       0.061481      0.089292    9.750734e-06   
24                  -0.000445       0.061526      0.089628    2.616577e-05   
25                  -0.001347       0.319737      1.916042    1.920165e-02   
26                  -0.001083       0.337366      1.966052    4.035160e-02   
27                  -0.001780       0.319737      1.915596    1.919710e-02   
28                  -0.001095       0.337366      1.966145    4.035328e-02   
29                  -0.000599       0.361187      1.978322    1.372939e-02   
30                  -0.001036       0.350701      1.971000    2.145446e-02   
31                  -0.000771       0.350701      1.971208    2.145937e-02   
32                  -0.000473       0.376939      1.991092    1.771377e-02   
33                  -0.000616       0.369770      1.991150    4.099574e-02   
34                  -0.000491       0.361187      1.978428    1.373164e-02   

         GreenReg  
0   volume**(1/3)  
1   volume**(1/3)  
2   volume**(1/3)  
3   volume**(1/3)  
4   volume**(1/3)  
5   volume**(1/3)  
6   volume**(1/3)  
7   volume**(1/3)  
8   volume**(1/3)  
9   volume**(1/3)  
10  volume**(1/3)  
11  volume**(1/3)  
12       fixed0p5  
13       fixed0p5  
14       fixed0p5  
15       fixed0p5  
16       fixed0p5  
17       fixed0p5  
18       fixed0p5  
19       fixed0p5  
20       fixed0p5  
21       fixed0p5  
22       fixed0p5  
23       fixed0p5  
24       fixed0p5  
25     fixed0p005  
26     fixed0p005  
27     fixed0p005  
28     fixed0p005  
29     fixed0p005  
30     fixed0p005  
31     fixed0p005  
32     fixed0p005  
33     fixed0p005  
34     fixed0p005  
>>> logAversusLogBcolorbyC(df.loc[df['GreenReg'] == 'volume**(1/3)'],'energyErrorGS','numberOfGridpoints','divideCriterion')

 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
>>> df
    domainSize  minDepth  maxDepth  numberOfCells  numberOfGridpoints  \
0           15         4         9          10144               90897   
1           15         4        10          10200               91849   
2           15         4        10           6784               61449   
3           15         4        10          16640              147473   
4           15         4        11          17032              151545   
5           15         4        11          10088               90929   
6           15         4        11          28624              249813   
7           15         4        12          30416              266189   
8           15         4        12          16864              150165   
9           15         4        11          53992              464505   
10          15         4        13          56288              483893   
11          15         4        13          29800              261429   
12          15         4        10          10200               91849   
13          15         4         9          10144               90897   
14          15         4        10          10200               91849   
15          15         4        10           6784               61449   
16          15         4        10          16640              147473   
17          15         4        11          17032              151545   
18          15         4        11          10088               90929   
19          15         4        11          28624              249813   
20          15         4        12          30416              266189   
21          15         4        12          16864              150165   
22          15         4        11          53992              464505   
23          15         4        13          56288              483893   
24          15         4        13          29800              261429   
25          15         4         9          10144               90897   
26          15         4        10          10200               91849   
27          15         4        10           6784               61449   
28          15         4        10          16640              147473   
29          15         4        11          17032              151545   
30          15         4        11          10088               90929   
31          15         4        11          28624              249813   
32          15         4        12          30416              266189   
33          15         4        12          16864              150165   
34          15         4        11          53992              464505   

    smoothingN  smoothingEps divideCriterion  divideParameter  energyResidual  \
0            0         1.000             LW1           1000.0        0.000001   
1            0         1.000             LW2           1000.0        0.000001   
2            0         1.000             LW3           1000.0        0.000001   
3            0         1.000             LW1           2000.0        0.000001   
4            0         1.000             LW2           2000.0        0.000001   
5            0         1.000             LW3           2000.0        0.000001   
6            0         1.000             LW1           4000.0        0.000001   
7            0         1.000             LW2           4000.0        0.000001   
8            0         1.000             LW3           4000.0        0.000001   
9            0         1.000             LW1           8000.0        0.000001   
10           0         1.000             LW2           8000.0        0.000001   
11           0         1.000             LW3           8000.0        0.000001   
12           0         1.000             LW2           1000.0        0.000001   
13           0         1.000             LW1           1000.0        0.000001   
14           0         1.000             LW2           1000.0        0.000001   
15           0         1.000             LW3           1000.0        0.000001   
16           0         1.000             LW1           2000.0        0.000001   
17           0         1.000             LW2           2000.0        0.000001   
18           0         1.000             LW3           2000.0        0.000001   
19           0         1.000             LW1           4000.0        0.000001   
20           0         1.000             LW2           4000.0        0.000001   
21           0         1.000             LW3           4000.0        0.000001   
22           0         1.000             LW1           8000.0        0.000001   
23           0         1.000             LW2           8000.0        0.000001   
24           0         1.000             LW3           8000.0        0.000001   
25           0         0.005             LW1           1000.0        0.000001   
26           0         0.005             LW2           1000.0        0.000001   
27           0         0.005             LW3           1000.0        0.000001   
28           0         0.005             LW1           2000.0        0.000001   
29           0         0.005             LW2           2000.0        0.000001   
30           0         0.005             LW3           2000.0        0.000001   
31           0         0.005             LW1           4000.0        0.000001   
32           0         0.005             LW2           4000.0        0.000001   
33           0         0.005             LW3           4000.0        0.000001   
34           0         0.005             LW1           8000.0        0.000001   

    energyErrorGS_analyticPsi  energyErrorGS  psiL2ErrorGS  psiLinfErrorGS  \
0                    0.001347      -0.001179      0.000117    1.392285e-07   
1                    0.001083      -0.001112      0.000076    7.236907e-08   
2                    0.001780      -0.001970      0.000081    1.308703e-07   
3                    0.001095      -0.000928      0.000135    1.423395e-07   
4                    0.000599      -0.000589      0.000068    3.207217e-08   
5                    0.001036      -0.001099      0.000118    9.634610e-08   
6                    0.000771      -0.000650      0.000114    4.946041e-08   
7                    0.000473      -0.000421      0.000070    1.932295e-08   
8                    0.000616      -0.000630      0.000073    3.414204e-08   
9                    0.000491      -0.000405      0.000076    2.076107e-08   
10                   0.000271      -0.000228      0.000040    5.558403e-09   
11                   1.998420      -0.000406      0.000072    2.525128e-08   
12                   1.995974       0.061347      0.091137    7.849253e-05   
13                  -0.001347       0.061008      0.092482    1.103454e-04   
14                  -0.001083       0.061347      0.091137    7.849253e-05   
15                  -0.001780       0.063191      0.097192    1.629038e-04   
16                  -0.001095       0.060338      0.089158    4.651185e-05   
17                  -0.000599       0.061234      0.089652    4.683014e-05   
18                  -0.001036       0.062180      0.091911    7.962988e-05   
19                  -0.000771       0.060581      0.088711    2.342316e-05   
20                  -0.000473       0.061309      0.089290    2.358815e-05   
21                  -0.000616       0.061484      0.089988    4.695751e-05   
22                  -0.000491       0.060831      0.088570    9.898562e-06   
23                  -0.000271       0.061481      0.089292    9.750734e-06   
24                  -0.000445       0.061526      0.089628    2.616577e-05   
25                  -0.001347       0.319737      1.916042    1.920165e-02   
26                  -0.001083       0.337366      1.966052    4.035160e-02   
27                  -0.001780       0.319737      1.915596    1.919710e-02   
28                  -0.001095       0.337366      1.966145    4.035328e-02   
29                  -0.000599       0.361187      1.978322    1.372939e-02   
30                  -0.001036       0.350701      1.971000    2.145446e-02   
31                  -0.000771       0.350701      1.971208    2.145937e-02   
32                  -0.000473       0.376939      1.991092    1.771377e-02   
33                  -0.000616       0.369770      1.991150    4.099574e-02   
34                  -0.000491       0.361187      1.978428    1.373164e-02   

         GreenReg  
0   volume**(1/3)  
1   volume**(1/3)  
2   volume**(1/3)  
3   volume**(1/3)  
4   volume**(1/3)  
5   volume**(1/3)  
6   volume**(1/3)  
7   volume**(1/3)  
8   volume**(1/3)  
9   volume**(1/3)  
10  volume**(1/3)  
11  volume**(1/3)  
12       fixed0p5  
13       fixed0p5  
14       fixed0p5  
15       fixed0p5  
16       fixed0p5  
17       fixed0p5  
18       fixed0p5  
19       fixed0p5  
20       fixed0p5  
21       fixed0p5  
22       fixed0p5  
23       fixed0p5  
24       fixed0p5  
25     fixed0p005  
26     fixed0p005  
27     fixed0p005  
28     fixed0p005  
29     fixed0p005  
30     fixed0p005  
31     fixed0p005  
32     fixed0p005  
33     fixed0p005  
34     fixed0p005  
>>> logAandBversusLogCcolorbyD(df,'energyErrorGS','energyErrorGS_analyticPsi','numberOfGridpoints','divideCriterion')

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 148
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 149
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 148
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 149
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
>>> 
 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
>>> logAandBversusLogCcolorbyD(df,'energyErrorGS','energyErrorGS_analyticPsi','numberOfGridpoints','divideCriterion')

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 148
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 149
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 148
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 149
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
>>> 
 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
>>> logAandBversusLogCcolorbyD(df,'energyErrorGS','energyErrorGS_analyticPsi','numberOfGridpoints','divideCriterion')

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 151
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 152
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 150
    group['logA'] = np.log10(np.abs(group[A]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 151
    group['logB'] = np.log10(np.abs(group[B]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

Warning (from warnings module):
  File "/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py", line 152
    group['logC'] = np.log10(np.abs(group[C]))
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
>>> 
 RESTART: /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/ErrorDataFrames.py 
>>>>>>> refs/remotes/eclipse_auto/master
