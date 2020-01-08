RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "I ${CYAN}love${NC} Stack Overflow"

# cd SinGAN_C_typeA
# echo -e "${CYAN}Start to train SinGAN_C_typeA.${NC}"
# python -W ignore seg_train.py
# cd ..

# date "+%H:%M:%S   %d/%m/%y"

# cd SinGAN_C_typeB
# echo -e "${CYAN}Start to train SinGAN_C_typeB.${NC}"
# python -W ignore seg_train.py
# cd ..

# date "+%H:%M:%S   %d/%m/%y"

# cd SinGAN_C_typeC
# echo -e "${CYAN}Start to train SinGAN_C_typeC.${NC}"
# python -W ignore seg_train.py
# cd ..

date "+%H:%M:%S   %d/%m/%y"

cd SinGAN_D_typeE
echo -e "${CYAN}Start to train SinGAN_D_typeE.${NC}"
python -W ignore seg_train.py
cd ..

date "+%H:%M:%S   %d/%m/%y"

cd SinGAN_D_typeF
echo -e "${CYAN}Start to train SinGAN_D_typeF.${NC}"
python -W ignore seg_train.py
cd ..

cd SinGAN_C_typeD
echo -e "${CYAN}Start to train SinGAN_C_typeD.${NC}"
python -W ignore seg_train.py
cd ..

date "+%H:%M:%S   %d/%m/%y"