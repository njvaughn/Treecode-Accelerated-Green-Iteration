/* tool functions for use by treecode routines */
#include <math.h>
#include "array.h"


double minval(double *x, int numels)
{
    int i;
    double min;

    min = x[0];

    for (i = 1; i < numels; i++) {
        if (min > x[i])
            min = x[i];
    }

    return min;
}




double maxval(double *x, int numels)
{
    int i;
    double max;

    max = x[0];

    for (i = 1; i < numels; i++) {
        if (max < x[i])
            max = x[i];
    }

    return max;
}




double sum(double *x, int numels)
{
    int i;
    double sum = 0.0;

    for (i = 0; i < numels; i++)
        sum = sum + x[i];

    return sum;
}




double max3(double a, double b, double c)
{
    double max;

    max = a;

    if (max < b) max = b;
    if (max < c) max = c;

    return max;
}




double min3(double a, double b, double c)
{
    double min;

    min = a;

    if (min > b) min = b;
    if (min > c) min = c;

    return min;
}
