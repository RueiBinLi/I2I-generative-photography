#!/bin/bash                                                                                                                                                                             
                                                                                                                                                                                        
# Define parameter lists                                                                                                                                                                
BOKEH_LIST="[2.44, 8.3, 10.1, 17.2, 24.0]"                                                                                                                                              
FOCAL_LIST="[25.0, 35.0, 45.0, 55.0, 65.0]"                                                                                                                                             
SHUTTER_LIST="[0.1, 0.3, 0.52, 0.7, 0.8]"                                                                                                                                               
COLOR_LIST="[5455.0, 5155.0, 5555.0, 6555.0, 7555.0]"                                                                                                                                   
                                                                                                                                                                                        
# Common settings                                                                                                                                                                       
INPUT_IMAGE="./input_image/my_park_photo.jpg"                                                                                                                                           
BASE_SCENE="A photo of a park with green grass and trees"                                                                                                                               
STRENGTH=0.8                                                                                                                                                                            
                                                                                                                                                                                        
# Arrays for iteration                                                                                                                                                                  
METHODS=("SDEdit" "DDIM_Inversion")                                                                                                                                                     
TYPES=("bokeh" "focal" "shutter" "color")                                                                                                                                               
                                                                                                                                                                                        
# Function to get list by type name                                                                                                                                                     
get_list() {                                                                                                                                                                            
    case $1 in
        "bokeh") echo "$BOKEH_LIST" ;;
        "focal") echo "$FOCAL_LIST" ;;
        "shutter") echo "$SHUTTER_LIST" ;;
        "color") echo "$COLOR_LIST" ;;
    esac
}

echo "Starting Ablation Study..."
echo "Total combinations: 2 methods * 4 types * 3 other types = 24 runs"

for method in "${METHODS[@]}"; do
    for type1 in "${TYPES[@]}"; do
        for type2 in "${TYPES[@]}"; do
            # Skip if types are the same
            if [ "$type1" == "$type2" ]; then
                continue
            fi

            echo "----------------------------------------------------------------"
            echo "Running: Method=$method, Stage1=$type1, Stage2=$type2"
             
            # Get values
            list1=$(get_list $type1)
            list2=$(get_list $type2)
             
            # Construct JSON string for multi_params
            # Note: We use python to ensure valid JSON formatting if needed, but simple string manipulation works here
            # Order matters: type1 first, then type2
            MULTI_PARAMS="{\"$type1\": $list1, \"$type2\": $list2}"
             
            # Define output directory specific to this combination
            # The python script appends the method name to this path
            OUTPUT_DIR="outputs/ablation/${type1}_${type2}"
             
            echo "Params: $MULTI_PARAMS"
            echo "Output Base: $OUTPUT_DIR"

            python unified_inference.py \
                --multi_params "$MULTI_PARAMS" \
                --input_image "$INPUT_IMAGE" \ 
                --base_scene "$BASE_SCENE" \
                --strength "$STRENGTH" \
                --method "$method" \
                --output_dir "$OUTPUT_DIR"

            echo "Finished: Method=$method, Stage1=$type1, Stage2=$type2"
        done
    done
done

echo "Ablation Study Completed!"