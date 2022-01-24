#include <stdio.h>
#include <stdlib.h>

int TFPN_CPP(unsigned char *truth, unsigned char *test, int gridsize, unsigned char TFPN_sel, unsigned char label);
void toOccup_CPP(unsigned char *sem_map, unsigned char *out, int gridsize);
int TP(unsigned char truth, unsigned char test, unsigned char label);
int TN(unsigned char truth, unsigned char test, unsigned char label);
int FP(unsigned char truth, unsigned char test, unsigned char label);
int FN(unsigned char truth, unsigned char test, unsigned char label);

extern "C" {
    void TFPN(unsigned char *truth, unsigned char *test, int gridsize, unsigned char TFPN_sel, unsigned char label)
    {
        TFPN_CPP(truth, test, gridsize, TFPN_sel, label);
    }
    void toOccup(unsigned char *sem_map, unsigned char *out, int gridsize)
    {
        toOccup_CPP(sem_map, out, gridsize);
    }
}

int main(int argc, char **argv)
{
    printf("Hello World!\n");

    return 0;
}


void toOccup_CPP(unsigned char *sem_map, unsigned char *out, int gridsize)
{
    int i = 0;
    for(i=0; i<gridsize*gridsize; i++)
        out[i] = sem_map[i] > 0b00000010;
}

int TFPN_CPP(unsigned char *truth, unsigned char *test, int gridsize, unsigned char TFPN_sel, unsigned char label)
{
    int i = 0;
    int cnt = 0;
    char comp_res = 0;

    for(i=0; i<gridsize*gridsize; i++)
    {
        switch(TFPN_sel)
        {
            case 0: 
                cnt += TP(truth[i], test[i], label);
                break;
            case 1: 
                cnt += TN(truth[i], test[i], label);
                break;
            case 2: 
                cnt += FP(truth[i], test[i], label);
                break;
            case 3: 
                cnt += FN(truth[i], test[i], label);
                break;
            default:
                printf("Wrong selector");
                break;
        }
    }
    return cnt;
}

int TP(unsigned char truth, unsigned char test, unsigned char label)
{
    if(label == 0x00)
        return (truth == test) && (test != 0x00);
    else
        return (truth == test) && (test == label);
}

int TN(unsigned char truth, unsigned char test, unsigned char label)
{
    if(label == 0x00)
        return (truth == test) && (test == 0x00);
    else
        return (truth == test) && (test != label);
}

int FP(unsigned char truth, unsigned char test, unsigned char label)
{
    if(label == 0x00)
        return (truth != test) && (test != 0x00);
    else
        return (truth != test) && (test == label);
}

int FN(unsigned char truth, unsigned char test, unsigned char label)
{
    if(label == 0x00)
        return (truth != test) && (test == 0x00);
    else
        return (truth != test) && (test != label);
}