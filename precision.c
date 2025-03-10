#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

typedef float item_t;

void read_items(item_t **items, size_t size, char *filename);

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "ERROR: Missing arguments\n");
        fprintf(stderr, "USAGE: '%s' <file1> <file2> <items>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *filename1 = argv[1];
    char *filename2 = argv[2];
    long size = strtol(argv[3], NULL, 10);
    if (size <= 0) {
        fprintf(stderr, "ERROR: Invalid number of items '%ld'\n", size);
        exit(EXIT_FAILURE);
    }

    printf("Comparing %s and %s with size %ld\n", filename1, filename2, size);

    item_t *items1 = NULL;
    item_t *items2 = NULL;

    read_items(&items1, size, filename1);
    read_items(&items2, size, filename2);

    float tot1 = 0.0;
    float tot2 = 0.0;
    for (size_t i = 0; i < (size_t) size; i++) {
        tot1 += items1[i];
        tot2 += items2[i];
    }

    float diff = tot2 - tot1;
    printf("Total 1: %f\n", tot1);
    printf("Total 2: %f\n", tot2);
    printf("Diff:    %f\n", diff);
    printf("Percent: %f%%\n", (diff/tot1)*100);

    free(items1);
    free(items2);

    return 0;
}

void read_items(item_t **items, size_t size, char *filename)
{
    printf("Opening file\n");
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "ERROR: Unable to open file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    printf("Allocating %ld memory\n", sizeof(item_t) * size);
    *items = malloc(sizeof(item_t) * size);
    if (items == NULL) {
        fprintf(stderr, "ERROR: Not enough memory\n");
        exit(EXIT_FAILURE);
    }

    printf("Reading file\n");
    size_t items_read = fread(*items, sizeof(item_t), size,  file);
    if (items_read != size) {
        fprintf(stderr, "ERROR: (%s) Only %zu items were read of %zu expected\n",
                filename, items_read, size);
        exit(EXIT_FAILURE);
    }

    printf("Closing file after reading '%zu'items\n", items_read);
    fclose(file);
}
