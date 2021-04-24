Step-1: 利用单通道纯净噪声借助仿真工具准备相应通道的散射噪声
	单通道纯净噪声：日常背景噪声、风噪、路边噪声、办公室噪声、商场噪声
	进入gen_diffuse_noise/Nonstationary_diffuse_noise 执行gen_diffuse_noise.m
	mic_pos = [0.0000, 0.000000; 0.0350, 0.000000; 0.0700, 0.000000; 0.1050, 0.000000; 0.1400, 0.000000; 0.1750, 0.000000]
	
	gen_diffuse_noise(noise_list, out_path, num_utts, M, mic_pos, gen_L);
	noise_list：单通道纯净噪声的文件列表
	out_path：输出目录
	num_utts：指定生成的散射噪声个数
	M：麦克风个数
	mic_pos：麦克风坐标位置
	gen_L：生成的句子长度，如果给-1，生成的句子长度和单通道纯净噪声的长度一致
	
Step-2：利用单声道纯净语音、单声道纯净干扰、散射噪声，利用仿真工具生成远场带噪麦克风阵列数据
		prepare_data_1mic.sh
		prepare_data_2mic.sh
		prepare_data_5mic.sh
		prepare_data_6mic.sh
	
	