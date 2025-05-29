import pandas as pd
from sqlalchemy import create_engine
import os


def criar_banco_de_dados_do_csv(csv_path, db_name, table_name_override=None):
    """
    Cria um banco de dados SQLite e uma tabela a partir de um arquivo CSV.

    Args:
        csv_path (str): Caminho para o arquivo CSV.
        db_name (str): Nome do arquivo do banco de dados SQLite a ser criado (ex: 'sus_data.db').
        table_name_override (str, optional): Nome específico para a tabela.
                                            Se None, o nome será inferido do CSV.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo CSV '{csv_path}' não encontrado. Verifique o caminho.")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV '{csv_path}': {e}")
        return

    if table_name_override:
        table_name = table_name_override
    else:
        # Gera um nome de tabela mais limpo a partir do nome do arquivo CSV
        base_filename = os.path.basename(csv_path)  # Ex: 'dados_sus3.csv'
        table_name = os.path.splitext(base_filename)[0]  # Ex: 'dados_sus3'
        table_name = table_name.replace('-', '_').replace(' ', '_')  # Garante que é um nome válido

    engine = create_engine(f"sqlite:///{db_name}")

    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Banco de dados '{db_name}' e tabela '{table_name}' criados/atualizados com sucesso.")
        print(f"Total de {len(df)} linhas carregadas para a tabela '{table_name}'.")
    except Exception as e:
        print(f"Erro ao salvar o DataFrame na tabela '{table_name}' do banco de dados SQL: {e}")

if __name__ == "__main__":
    # Configurações
    csv_file = 'data/dados_sus3.csv'  # Ajuste o caminho se necessário
    database_file = 'sus_data.db'

    # Você pode opcionalmente definir um nome de tabela específico aqui,
    # ou deixar None para que seja inferido do nome do arquivo CSV.
    # Ex: nome_da_tabela = "minha_tabela_sus"
    nome_da_tabela = None

    print(f"Iniciando o processo de criação do banco de dados a partir de '{csv_file}'...")
    criar_banco_de_dados_do_csv(csv_file, database_file, nome_da_tabela)
    print("Processo finalizado.")