import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

MODEL_ID = "defog/sqlcoder-7b-2"
#https://huggingface.co/defog/sqlcoder-7b

def carregar_llm_langchain(use_gpu_if_available=True):
    """
    > Carrega o modelo SQLCoder usando HuggingFacePipeline do Langchain.
    > usa GPU e quantização
    > Caso contrário, tenta fallback para CPU (sem quantização de 4 bits).
    """
    print(f"Iniciando configuração do LLM via Langchain: {MODEL_ID}")
    device_to_use = -1
    model_kwargs_config = {'trust_remote_code': True}
    pipeline_kwargs_config = {}
    pipeline_kwargs_config['max_new_tokens'] = 512

    if use_gpu_if_available and torch.cuda.is_available():
        print("Tentando configurar para GPU com quantização de 4 bits...")
        try:
            torch.zeros(1).cuda()  # Teste rápido
            print("CUDA parece estar funcional.")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs_config['quantization_config'] = quantization_config
            model_kwargs_config['torch_dtype'] = torch.bfloat16
            model_kwargs_config['device_map'] = "auto"  #alocação na GPU

            print("Configurado para GPU com quantização.")
        except Exception as e:
            print(f"Falha ao configurar para GPU (erro: {e}). Tentando fallback para CPU.")
            print("O erro 'libcudart.so.11.0' provavelmente persistirá aqui se a causa raiz não for resolvida.")
            model_kwargs_config = {'trust_remote_code': True, 'torch_dtype': torch.float32}  # Reset para CPU
            device_to_use = -1  # CPU
            pipeline_kwargs_config['device'] = device_to_use
            print("Configurado para CPU.")
    else:
        print("CUDA não disponível ou use_gpu_if_available=False. Configurando para CPU.")
        model_kwargs_config = {'trust_remote_code': True, 'torch_dtype': torch.float32}
        device_to_use = -1  # CPU
        pipeline_kwargs_config['device'] = device_to_use

    try:
        print(f"Carregando LLM com model_kwargs: {model_kwargs_config}")
        print(f"Argumento 'device' para o pipeline (se aplicável): {pipeline_kwargs_config.get('device')}")

        # Se device_map está em model_kwargs, o parâmetro device do HuggingFacePipeline pode não ser necessário
        # ou pode precisar ser consistente. Se device_map não for usado, device no HF Pipeline é importante.
        if "device_map" in model_kwargs_config:
            # Deixe o device_map cuidar disso. Não passe 'device' para HuggingFacePipeline diretamente.
            if 'device' in pipeline_kwargs_config:  # Evitar passar os dois
                del pipeline_kwargs_config['device']

        llm = HuggingFacePipeline.from_model_id(
            model_id=MODEL_ID,
            task="text-generation",  # SQLCoder é um modelo de geração de texto
            model_kwargs=model_kwargs_config,
            pipeline_kwargs=pipeline_kwargs_config  # Passa o device para o pipeline se não usar device_map
        )

        print(f"\nLLM ({MODEL_ID}) carregado com sucesso usando Langchain HuggingFacePipeline!")
        if "device_map" in model_kwargs_config and hasattr(llm.pipeline.model, 'hf_device_map'):
            print(f"Modelo distribuído em: {llm.pipeline.model.hf_device_map}")
        elif hasattr(llm.pipeline, 'device'):
            print(f"Pipeline rodando no dispositivo: {llm.pipeline.device}")
        return llm

    except Exception as e:
        print(f"Ocorreu um erro crítico ao carregar o LLM via Langchain: {e}")
        if "libcudart" in str(e):
            print(
                "Este erro está relacionado à sua instalação do CUDA. Por favor, revise as etapas de diagnóstico do CUDA da mensagem anterior.")
        return None


if __name__ == "__main__":
    print("--- Teste de Configuração do LLM SQLCoder via Langchain ---")
    # Tente usar GPU por padrão. Se falhar devido ao CUDA, a função tentará CPU.
    # Se você sabe que o CUDA não funciona, pode passar use_gpu_if_available=False
    llm_instance = carregar_llm_langchain(use_gpu_if_available=True)

    if llm_instance:
        print("\nConcluído: Instância do LLM Langchain foi criada.")
        print("Se não houve erros e o dispositivo correto foi reportado, o modelo está pronto para ser usado.")

    else:
        print("\nFalha: Não foi possível criar a instância do LLM Langchain.")