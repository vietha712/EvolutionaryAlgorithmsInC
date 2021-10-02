# Utilities
PROJ := main

# Problem
PROBLEM := planar_truss_52bars

#Optimization algorithm
ALGO := parallel_aeDE

#LIB
LIB := matrix

#Paths
OUTPUT_PATH := _build
INC_PATH :=
SRC_PATH := src

# Includes
INC := -I$(INC_PATH)
INCLUDES := $(INC_PATH)

# Compiler
CC := gcc
CC_OPT := -Wall -pedantic -O3 -funroll-loops -fopenmp

# Search paths
vpath %.c $(SRC_PATH)

#OBJECT FILES
OBJ_FILES := $(OUTPUT_PATH)/$(PROBLEM).o $(OUTPUT_PATH)/$(LIB).o $(OUTPUT_PATH)/$(ALGO).o

# Commands

#OUTPUT_PATH/%.o: %.c
#	$(CC) -c $(CC_OPT) $(INCLUDES) $< -o $@

build: $(PROJ).c $(OBJ_FILES) 
	@echo "Compiling $(PROJ).c"
	@$(CC) $(CC_OPT) $(OBJ_FILES) $(PROJ).c -o $(OUTPUT_PATH)/$(PROJ).exe -lm

$(OUTPUT_PATH)/$(LIB).o: $(LIB).c
	@$(CC) -c $(CC_OPT) $< -o $(OUTPUT_PATH)/$(LIB).o

$(OUTPUT_PATH)/$(ALGO).o: $(ALGO).c
	@$(CC) -c $(CC_OPT) $< -o $(OUTPUT_PATH)/$(ALGO).o

$(OUTPUT_PATH)/$(PROBLEM).o: $(PROBLEM).c build-dir
	@$(CC) -c $(CC_OPT) $< -o $(OUTPUT_PATH)/$(PROBLEM).o

build_test:
	gcc -Wall -g -pedantic -O3 -o test_main.exe test_main.c planar_truss_72bars_1d.c matrix1D.c -lm

build_de_ten_bars:
	gcc -Wall -pedantic -O -o de de.c planar_struss_10bars.c matrix.c -lm

build_de_five_two_bars:
	gcc -Wall -pedantic -O -o de de.c planar_truss_52bars.c matrix.c -lm

build_de_seven_two_bars:
	gcc -Wall -pedantic -O -o de de.c planar_truss_72bars.c matrix.c -lm

run_truss:
	./de -N 30 -G 2000 -C 0.6 -F 1.0

build-dir:
	@mkdir -p $(OUTPUT_PATH)

clean:
	@echo "Cleaning up"
	@rm -rf  $(OUTPUT_PATH)

debug:
	gdb $(OUTPUT_PATH)/$(PROJ).exe

run:
	@echo "Executing program"
	@echo "Outcome can be found below"
	@echo ""
	@$(OUTPUT_PATH)/$(PROJ).exe