# Utilities
PROJ := main

# Problem
PROBLEM := schwefel_20d

# Algorithms
ALGO := ae_de

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
OBJ_FILES := $(OUTPUT_PATH)/$(PROBLEM).o $(OUTPUT_PATH)/$(ALGO).o

# Commands

#OUTPUT_PATH/%.o: %.c
#	$(CC) -c $(CC_OPT) $(INCLUDES) $< -o $@

build: $(PROJ).c $(OBJ_FILES) 
	@echo "Compiling $(PROJ).c"
	@$(CC) $(CC_OPT) $(OBJ_FILES) $(PROJ).c -o $(OUTPUT_PATH)/$(PROJ).exe -lm

$(OUTPUT_PATH)/$(PROBLEM).o: $(PROBLEM).c build-dir
	@$(CC) -c $(CC_OPT) $< -o $(OUTPUT_PATH)/$(PROBLEM).o

$(OUTPUT_PATH)/$(ALGO).o: $(ALGO).c build-dir
	@$(CC) -c $(CC_OPT) $< -o $(OUTPUT_PATH)/$(ALGO).o

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