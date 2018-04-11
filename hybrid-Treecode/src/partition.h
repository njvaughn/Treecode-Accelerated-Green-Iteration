#ifndef H_PARTITION_H
#define H_PARTITION_H

/* 
 * declaration of partition functions
 *
 * partition determines the index MIDIND, after partitioning in place the arrays a, b, c,
 * and q, such that a(ibeg:midind) <= val and a(midind+1:iend) > val. If on entry, ibeg >
 * iend, or a(ibeg:iend) > val then midind is returned as ibeg-1.
 */

void cp_partition(double *a, double *b, double *c, int *indarr,
                  int ibeg, int iend, double val, int *midind);

void pc_partition(double *a, double *b, double *c, double *d, int *indarr,
                  int ibeg, int iend, double val, int *midind);


#endif /* H_PARTITION_H */
