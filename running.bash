echo "Pix2Pix"
python running.py --input_picture ulm_000000_000019_gtFine_color.png --pix2pix_path pytorch_pix2pix/checkpoints/cityUnetmix2/latest_net_G.pth --source_picture ulm_000000_000019_leftImg8bit.png
display ulm_000000_000019_gtFine_color_fake.png &
sleep 10

echo "SinGAN_C_typeA"
python -W ignore running.py --input_picture ulm_000000_000019_gtFine_color.png --singan_model_dir SinGAN_C_typeA --singan_model cityscapes --source_picture ulm_000000_000019_leftImg8bit.png --mode B --manualSeed 6950
display SinGAN_Output/cityscapes_A/ulm_000000_000019_gtFine_color.png &
sleep 5

echo "SinGAN_C_typeB"
python -W ignore running.py --input_picture ulm_000000_000019_gtFine_color.png --singan_model_dir SinGAN_C_typeB --singan_model cityscapes --source_picture ulm_000000_000019_leftImg8bit.png --mode B --manualSeed 6950
display SinGAN_Output/cityscapes_B/ulm_000000_000019_gtFine_color.png &
sleep 5

echo "SinGAN_C_typeC"
python -W ignore running.py --input_picture ulm_000000_000019_gtFine_color.png --singan_model_dir SinGAN_C_typeC --singan_model cityscapes --source_picture ulm_000000_000019_leftImg8bit.png --mode B --manualSeed 6950
display SinGAN_Output/cityscapes_C/ulm_000000_000019_gtFine_color.png &
sleep 5

echo "SinGAN_C_typeD"
python -W ignore running.py --input_picture ulm_000000_000019_gtFine_color.png --singan_model_dir SinGAN_C_typeD --singan_model cityscapes --source_picture ulm_000000_000019_leftImg8bit.png --mode B --manualSeed 6950 --size 256
display SinGAN_Output/cityscapes_D/ulm_000000_000019_gtFine_color.png &
sleep 5

echo "SinGAN_D_typeE"
python -W ignore running.py --input_picture ulm_000000_000019_gtFine_color.png --singan_model_dir SinGAN_D_typeE --singan_model cityscapes --source_picture ulm_000000_000019_leftImg8bit.png --mode B --manualSeed 6950
display SinGAN_Output/cityscapes_E/ulm_000000_000019_gtFine_color.png &
sleep 5

echo "SinGAN_D_typeF"
python -W ignore running.py --input_picture ulm_000000_000019_gtFine_color.png --singan_model_dir SinGAN_D_typeF --singan_model cityscapes --source_picture ulm_000000_000019_leftImg8bit.png --mode B --manualSeed 6950
display SinGAN_Output/cityscapes_F/ulm_000000_000019_gtFine_color.png &
sleep 5
