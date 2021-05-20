# Utilities
PROJ := test_main

# Problem
PROBLEM := planar_truss_10bars

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
CC_OPT := -Wall -pedantic -O3 -funroll-loops

# Search paths
vpath %.c $(SRC_PATH)

#OBJECT FILES
OBJ_FILES := $(OUTPUT_PATH)/$(PROBLEM).o $(OUTPUT_PATH)/$(LIB).o

# Commands

#OUTPUT_PATH/%.o: %.c
#	$(CC) -c $(CC_OPT) $(INCLUDES) $< -o $@

build: $(PROJ).c $(OBJ_FILES) 
	@echo "Compiling $(PROJ).c"
	@$(CC) $(CC_OPT) $(OBJ_FILES) $(PROJ).c -o $(OUTPUT_PATH)/$(PROJ).exe -lm

$(OUTPUT_PATH)/$(LIB).o: $(LIB).c
	@$(CC) -c $(CC_OPT) $< -o $(OUTPUT_PATH)/$(LIB).o

$(OUTPUT_PATH)/$(PROBLEM).o: $(PROBLEM).c build-dir
	@$(CC) -c $(CC_OPT) $< -o $(OUTPUT_PATH)/$(PROBLEM).o

build_test:
	gcc -Wall -pedantic -O -o test_main.exe test_main.c planar_struss_10bars.c matrix.c -lm

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