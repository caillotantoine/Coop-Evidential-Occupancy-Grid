#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

void set_inter(const char *A, const char *B, char *out);
void set_union(const char *A, const char *B, char *out);
bool set_cmp(const char *A, const char *B);

extern "C" {}

using namespace std;

int main(int argc, char **argv)
{
    char FE[8][4] = {"O", "V", "P", "T", "VP", "VT", "PT", "VPT"};
    char out[4] = {0};
    set_union(FE[4], FE[5], out);

    printf("%s\n", out);

    cout << set_cmp("VP", "PVT") << endl;

    return 0;
}

void set_inter(const char *A, const char *B, char *out)
{
    int i = 0, j = 0;
    for(i=0; i<strlen(A); i++)
    {
        if(strchr(B, A[i]) != NULL)
        {
            out[j] = A[i];
            j++;
        }
    }   
}

void set_union(const char *A, const char *B, char *out)
{
    int i = 0, j = strlen(B);
    out = strcpy(out, B);
    for(i=0; i<strlen(A); i++)
    {
        if(strchr(B, A[i]) == NULL)
        {
            out[j] = A[i];
            j++;
        }
    }   
}

bool set_cmp(const char *A, const char *B)
{
    int i = 0;
    if(strlen(A) != strlen(B))
        return false;

    for(i = 0; i<strlen(A); i++)
    {
        if(strchr(B, A[i]) == NULL)
            return false;
    }
    return true;
}