from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_deepseek_model():
    """載入DeepSeek模型"""
    print("正在載入DeepSeek模型...")
    
    # 選擇模型 - 可以根據需要更換
    model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
    
    try:
        # 載入tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 修復pad token問題
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 載入模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # 使用半精度節省VRAM
            device_map="auto",          # 自動分配GPU
            trust_remote_code=True
        )
        
        # 設定模型的pad token
        model.config.pad_token_id = tokenizer.pad_token_id
        
        print(f"模型載入完成！使用設備: {model.device}")
        print(f"GPU記憶體使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"Pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return None, None

def generate_response(model, tokenizer, user_input, max_length=512):
    """生成回應"""
    try:
        # 格式化為聊天格式
        messages = [
            {"role": "user", "content": user_input}
        ]
        
        # 使用tokenizer的chat template
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 編碼輸入並創建attention mask
        encoded = tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        input_ids = encoded['input_ids'].to(model.device)
        attention_mask = encoded['attention_mask'].to(model.device)
        
        # 生成回應
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # 明確提供attention mask
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # 解碼回應（只取新生成的部分）
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        return f"生成回應時發生錯誤: {e}"

def chat_with_deepseek():
    """主要聊天循環"""
    print("=== DeepSeek 互動式聊天程式 ===")
    print("載入中，請稍候...")
    
    # 載入模型
    model, tokenizer = load_deepseek_model()
    
    if model is None or tokenizer is None:
        print("無法載入模型，程式結束。")
        return
    
    print("\n模型已準備完成！")
    print("輸入您的問題或指令，輸入 'quit' 或 'exit' 結束程式")
    print("=" * 50)
    
    while True:
        try:
            # 獲取使用者輸入
            user_input = input("\n您: ").strip()
            
            # 檢查退出指令
            if user_input.lower() in ['quit', 'exit', '退出', '結束']:
                print("再見！")
                break
            
            # 檢查空輸入
            if not user_input:
                print("請輸入一些內容...")
                continue
            
            # 生成回應
            print("\nDeepSeek: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            
            # 顯示記憶體使用情況
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"\n[GPU記憶體使用: {memory_used:.2f} GB]")
                
        except KeyboardInterrupt:
            print("\n\n程式被中斷，再見！")
            break
        except Exception as e:
            print(f"\n發生錯誤: {e}")
            continue

def quick_test():
    """快速測試函數"""
    print("=== 快速測試模式 ===")
    
    model, tokenizer = load_deepseek_model()
    if model is None:
        return
    
    # 測試問題
    test_prompts = [
        "寫一個Python函數來計算費波那契數列",
        "解釋什麼是遞迴",
        "幫我寫一個排序演算法"
    ]
    
    for prompt in test_prompts:
        print(f"\n測試問題: {prompt}")
        print("-" * 30)
        response = generate_response(model, tokenizer, prompt)
        print(f"回應: {response}")
        print("=" * 50)

if __name__ == "__main__":
    # 檢查CUDA是否可用
    if torch.cuda.is_available():
        print(f"檢測到GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("警告: 未檢測到CUDA，將使用CPU（速度較慢）")
    
    # 詢問使用者要執行哪種模式
    print("\n選擇模式:")
    print("1. 互動式聊天")
    print("2. 快速測試")
    
    try:
        choice = input("請選擇 (1 或 2): ").strip()
        
        if choice == "1":
            chat_with_deepseek()
        elif choice == "2":
            quick_test()
        else:
            print("無效選擇，啟動互動式聊天...")
            chat_with_deepseek()
            
    except KeyboardInterrupt:
        print("\n程式結束。")