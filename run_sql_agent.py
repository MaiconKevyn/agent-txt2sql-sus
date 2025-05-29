import os
import re
# import json
# import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
# from sqlalchemy.exc import OperationalError
from database import criar_banco_de_dados_do_csv
from llm_config import carregar_llm_langchain
from langchain_community.utilities import SQLDatabase

# Configurações
CSV_FILE_PATH = 'data/dados_sus3.csv'
DB_FILE_NAME = 'sus_data.db'
DB_URI = f"sqlite:///{DB_FILE_NAME}"
TABLE_NAME = 'dados_sus'

class QueryComponent(Enum):
    """Tipos de componentes de consulta"""
    SELECT = "select"
    WHERE = "where"
    GROUP_BY = "group_by"
    ORDER_BY = "order_by"
    HAVING = "having"
    LIMIT = "limit"

class AggregationType(Enum):
    """Tipos de agregação"""
    COUNT = "COUNT"
    AVG = "AVG"
    SUM = "SUM"
    MAX = "MAX"
    MIN = "MIN"
    DISTINCT_COUNT = "COUNT(DISTINCT {})"

@dataclass
class QueryCondition:
    """Representa uma condição WHERE"""
    column: str
    operator: str
    value: Union[str, int, float]
    logical_connector: str = "AND"  # AND, OR

    def to_sql(self) -> str:
        if isinstance(self.value, str):
            return f"{self.column} {self.operator} '{self.value}'"
        return f"{self.column} {self.operator} {self.value}"

@dataclass
class SemanticQuery:
    """Representa uma consulta semanticamente analisada"""
    target_columns: List[str] = field(default_factory=list)
    aggregations: Dict[str, str] = field(default_factory=dict)  # column -> aggregation_type
    conditions: List[QueryCondition] = field(default_factory=list)
    group_by_columns: List[str] = field(default_factory=list)
    order_by: Optional[Tuple[str, str]] = None  # (column, direction)
    limit: Optional[int] = None
    response_context: Dict[str, Any] = field(default_factory=dict)

class SemanticAnalyzer:
    """Analisador semântico avançado para consultas em linguagem natural"""

    def __init__(self):
        self.column_mappings = self._initialize_column_mappings()
        self.operator_mappings = self._initialize_operator_mappings()
        self.aggregation_mappings = self._initialize_aggregation_mappings()
        self.value_mappings = self._initialize_value_mappings()

    def _initialize_column_mappings(self) -> Dict[str, str]:
        """Mapeia termos em linguagem natural para colunas SQL"""
        return {
            # Idade
            'idade': 'IDADE',
            'idades': 'IDADE',
            'idoso': 'IDADE',
            'idosos': 'IDADE',
            'criança': 'IDADE',
            'crianças': 'IDADE',
            'adolescente': 'IDADE',
            'adulto': 'IDADE',

            # UTI
            'uti': 'UTI_MES_TO',
            'unidade terapia intensiva': 'UTI_MES_TO',
            'dias uti': 'UTI_MES_TO',
            'permanência uti': 'UTI_MES_TO',
            'tempo uti': 'UTI_MES_TO',

            # Diagnóstico
            'diagnóstico': 'DIAG_PRINC',
            'diagnostico': 'DIAG_PRINC',
            'diagnósticos': 'DIAG_PRINC',
            'doença': 'DIAG_PRINC',
            'doenças': 'DIAG_PRINC',
            'cid': 'DIAG_PRINC',

            # Geografia
            'cidade': 'CIDADE_RESIDENCIA_PACIENTE',
            'cidades': 'CIDADE_RESIDENCIA_PACIENTE',
            'município': 'CIDADE_RESIDENCIA_PACIENTE',
            'estado': 'UF_RESIDENCIA_PACIENTE',
            'estados': 'UF_RESIDENCIA_PACIENTE',
            'uf': 'UF_RESIDENCIA_PACIENTE',

            # Sexo
            'sexo': 'SEXO',
            'gênero': 'SEXO',
            'genero': 'SEXO',
            'masculino': 'SEXO',
            'feminino': 'SEXO',
            'homem': 'SEXO',
            'mulher': 'SEXO',

            # Valores
            'valor': 'VAL_TOT',
            'valores': 'VAL_TOT',
            'custo': 'VAL_TOT',
            'custos': 'VAL_TOT',
            'gasto': 'VAL_TOT',
            'gastos': 'VAL_TOT',
            'procedimento': 'VAL_TOT',

            # Mortalidade
            'morte': 'MORTE',
            'mortes': 'MORTE',
            'óbito': 'MORTE',
            'obito': 'MORTE',
            'óbitos': 'MORTE',
            'obitos': 'MORTE',
            'mortalidade': 'MORTE',

            # Datas
            'internação': 'DT_INTER',
            'internacao': 'DT_INTER',
            'saída': 'DT_SAIDA',
            'saida': 'DT_SAIDA',
            'alta': 'DT_SAIDA',
        }

    def _initialize_operator_mappings(self) -> Dict[str, str]:
        """Mapeia operadores em linguagem natural para SQL"""
        return {
            # Comparações numéricas
            'maior que': '>',
            'maior do que': '>',
            'maiores que': '>',
            'maiores do que': '>',
            'acima de': '>',
            'superior a': '>',
            'mais de': '>',
            'mais que': '>',

            'menor que': '<',
            'menor do que': '<',
            'menores que': '<',
            'menores do que': '<',
            'abaixo de': '<',
            'inferior a': '<',
            'menos de': '<',
            'menos que': '<',

            'maior ou igual': '>=',
            'maior igual': '>=',
            'pelo menos': '>=',
            'no mínimo': '>=',

            'menor ou igual': '<=',
            'menor igual': '<=',
            'no máximo': '<=',
            'até': '<=',

            'igual a': '=',
            'igual': '=',
            'exatamente': '=',

            'diferente de': '!=',
            'diferente': '!=',
            'não igual': '!=',

            # Operadores textuais
            'contém': 'LIKE',
            'contem': 'LIKE',
            'inclui': 'LIKE',
            'como': 'LIKE',

            # Ranges
            'entre': 'BETWEEN',
            'de': 'BETWEEN',  # "de X a Y"
        }

    def _initialize_aggregation_mappings(self) -> Dict[str, str]:
        """Mapeia funções de agregação"""
        return {
            'média': 'AVG',
            'media': 'AVG',
            'médio': 'AVG',
            'medio': 'AVG',
            'média de': 'AVG',

            'total': 'SUM',
            'soma': 'SUM',
            'somatória': 'SUM',
            'somatoria': 'SUM',

            'quantidade': 'COUNT',
            'quantos': 'COUNT',
            'número': 'COUNT',
            'numero': 'COUNT',
            'count': 'COUNT',

            'máximo': 'MAX',
            'maximo': 'MAX',
            'maior': 'MAX',
            'max': 'MAX',

            'mínimo': 'MIN',
            'minimo': 'MIN',
            'menor': 'MIN',
            'min': 'MIN',

            'distintos': 'COUNT(DISTINCT {})',
            'únicos': 'COUNT(DISTINCT {})',
            'unicos': 'COUNT(DISTINCT {})',
            'diferentes': 'COUNT(DISTINCT {})',
        }

    def _initialize_value_mappings(self) -> Dict[str, Any]:
        """Mapeia valores específicos para códigos"""
        return {
            # Sexo
            'masculino': 1,
            'homem': 1,
            'homens': 1,
            'feminino': 3,
            'mulher': 3,
            'mulheres': 3,

            # Mortalidade
            'sim': 1,
            'não': 0,
            'nao': 0,
            'verdadeiro': 1,
            'falso': 0,
            'true': 1,
            'false': 0,
        }

    def analyze_question(self, question: str) -> SemanticQuery:
        """Analisa semanticamente uma pergunta e retorna SemanticQuery"""
        question_lower = question.lower()
        query = SemanticQuery()

        # 1. Identificar agregações
        self._extract_aggregations(question_lower, query)

        # 2. Identificar colunas alvo
        self._extract_target_columns(question_lower, query)

        # 3. Identificar condições WHERE
        self._extract_conditions(question_lower, query)

        # 4. Identificar agrupamentos
        self._extract_groupings(question_lower, query)

        # 5. Identificar ordenação e limite
        self._extract_ordering_and_limits(question_lower, query)

        # 6. Adicionar contexto para resposta
        self._add_response_context(question_lower, query)

        return query

    def _extract_aggregations(self, question: str, query: SemanticQuery):
        """Extrai funções de agregação da pergunta"""
        # CORRIGIDO: Melhor detecção de agregações
        for term, agg_type in self.aggregation_mappings.items():
            if term in question:
                # Casos especiais para COUNT
                if term in ['quantos', 'total', 'numero', 'quantidade']:
                    # Verificar se é contagem de registros/casos
                    if any(word in question for word in ['casos', 'registros', 'pacientes', 'pessoas']):
                        query.aggregations['*'] = 'COUNT'
                        query.response_context['aggregation_term'] = term
                        break

                # Para outras agregações, encontrar a coluna relacionada
                for col_term, col_name in self.column_mappings.items():
                    if col_term in question:
                        if '{}' in agg_type:
                            query.aggregations[col_name] = agg_type.format(col_name)
                        else:
                            query.aggregations[col_name] = agg_type
                        query.response_context['aggregation_term'] = term
                        break

                # Se não encontrou coluna específica mas tem agregação, usar contagem
                if not query.aggregations and term in ['quantos', 'total', 'numero', 'quantidade']:
                    query.aggregations['*'] = 'COUNT'
                    query.response_context['aggregation_term'] = term

                if query.aggregations:
                    break

    def _extract_target_columns(self, question: str, query: SemanticQuery):
        """Extrai colunas alvo da consulta"""
        # Se já tem agregação, não precisa de colunas adicionais no SELECT
        if query.aggregations:
            return

        # Buscar colunas mencionadas na pergunta
        mentioned_columns = []
        for term, column in self.column_mappings.items():
            if term in question and column not in mentioned_columns:
                mentioned_columns.append(column)

        query.target_columns = mentioned_columns

    def _extract_conditions(self, question: str, query: SemanticQuery):
        """Extrai condições WHERE complexas"""
        # CORRIGIDO: Evitar duplicação de condições de idade
        age_condition_found = False

        # Padrões para extrair condições numéricas
        age_patterns = [
            r'idade[s]?\s+(maior[es]?\s+que|maiores?\s+do\s+que|acima\s+de|superior\s+a|mais\s+de)\s+(\d+)',
            r'idade[s]?\s+(menor[es]?\s+que|menores?\s+do\s+que|abaixo\s+de|inferior\s+a|menos\s+de)\s+(\d+)',
            r'idade[s]?\s+(maior\s+ou\s+igual|maior\s+igual|pelo\s+menos|no\s+mínimo)\s+(\d+)',
            r'idade[s]?\s+(menor\s+ou\s+igual|menor\s+igual|no\s+máximo|até)\s+(\d+)',
            r'idade[s]?\s+(igual\s+a?|exatamente)\s+(\d+)',
            r'idade[s]?\s+entre\s+(\d+)\s+e\s+(\d+)',
            r'(\d+)\s+anos?\s+ou\s+mais',
            r'(\d+)\s+anos?\s+ou\s+menos',
            r'mais\s+de\s+(\d+)\s+anos?',
            r'menos\s+de\s+(\d+)\s+anos?',
            r'acima\s+de\s+(\d+)\s+anos?',
            r'abaixo\s+de\s+(\d+)\s+anos?',
        ]

        # Processar padrões de idade
        for pattern in age_patterns:
            match = re.search(pattern, question)
            if match and not age_condition_found:
                groups = match.groups()
                if 'maior' in groups[0] or 'acima' in groups[0] or 'superior' in groups[0] or 'mais' in groups[0]:
                    operator = '>'
                elif 'menor' in groups[0] or 'abaixo' in groups[0] or 'inferior' in groups[0] or 'menos' in groups[0]:
                    operator = '<'
                elif 'maior ou igual' in groups[0] or 'pelo menos' in groups[0] or 'no mínimo' in groups[0]:
                    operator = '>='
                elif 'menor ou igual' in groups[0] or 'no máximo' in groups[0] or 'até' in groups[0]:
                    operator = '<='
                elif 'igual' in groups[0] or 'exatamente' in groups[0]:
                    operator = '='
                elif 'entre' in groups[0]:
                    # Caso especial para BETWEEN
                    value1, value2 = int(groups[1]), int(groups[2])
                    condition = QueryCondition('IDADE', 'BETWEEN', f"{value1} AND {value2}")
                    query.conditions.append(condition)
                    age_condition_found = True
                    break
                else:
                    operator = '>'  # default

                value = int(groups[1]) if len(groups) > 1 else int(groups[0])
                condition = QueryCondition('IDADE', operator, value)
                query.conditions.append(condition)
                age_condition_found = True
                break

        # Padrões especiais para "ou mais/menos" no final - apenas se não encontrou antes
        if not age_condition_found:
            age_suffix_patterns = [
                (r'(\d+)\s+anos?\s+ou\s+mais', '>='),
                (r'(\d+)\s+anos?\s+ou\s+menos', '<='),
                (r'mais\s+de\s+(\d+)\s+anos?', '>'),
                (r'menos\s+de\s+(\d+)\s+anos?', '<'),
                (r'acima\s+de\s+(\d+)\s+anos?', '>'),
                (r'abaixo\s+de\s+(\d+)\s+anos?', '<'),
            ]

            for pattern, operator in age_suffix_patterns:
                match = re.search(pattern, question)
                if match:
                    value = int(match.group(1))
                    condition = QueryCondition('IDADE', operator, value)
                    query.conditions.append(condition)
                    break

        # Condições para outras colunas (sexo, cidade, etc.)
        self._extract_categorical_conditions(question, query)

        # Condições para valores numéricos (UTI, valores)
        self._extract_numeric_conditions(question, query)

    def _extract_categorical_conditions(self, question: str, query: SemanticQuery):
        """Extrai condições para variáveis categóricas"""
        # Sexo
        if 'masculino' in question or 'homem' in question or 'homens' in question:
            condition = QueryCondition('SEXO', '=', 1)
            query.conditions.append(condition)
        elif 'feminino' in question or 'mulher' in question or 'mulheres' in question:
            condition = QueryCondition('SEXO', '=', 3)
            query.conditions.append(condition)

        # Cidade específica - CORRIGIDO: evitar palavras como "casos"
        # Lista de palavras que NÃO são cidades
        non_city_words = {'casos', 'total', 'para', 'com', 'sem', 'mais', 'menos',
                          'idades', 'anos', 'dias', 'uti', 'obito', 'morte', 'valor',
                          'media', 'maximo', 'minimo', 'pacientes', 'registros'}

        city_patterns = [
            r'em\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',  # em Porto Alegre
            r'na\s+cidade\s+de\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
            r'no\s+município\s+de\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
            r'cidade\s+de\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
        ]

        for pattern in city_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                city_candidate = match.group(1).lower().strip()
                # Verificar se não é uma palavra comum que não é cidade
                if city_candidate not in non_city_words and len(city_candidate) > 2:
                    city = city_candidate.upper()
                    condition = QueryCondition('CIDADE_RESIDENCIA_PACIENTE', 'LIKE', f'%{city}%')
                    query.conditions.append(condition)
                    break

        # Mortalidade
        if any(term in question for term in ['com óbito', 'com obito', 'que morreram', 'mortes']):
            condition = QueryCondition('MORTE', '=', 1)
            query.conditions.append(condition)
        elif any(term in question for term in ['sem óbito', 'sem obito', 'que sobreviveram', 'vivos']):
            condition = QueryCondition('MORTE', '=', 0)
            query.conditions.append(condition)

    def _extract_numeric_conditions(self, question: str, query: SemanticQuery):
        """Extrai condições para variáveis numéricas (além de idade)"""
        # UTI
        uti_patterns = [
            r'uti\s+(maior[es]?\s+que|mais\s+de)\s+(\d+)',
            r'uti\s+(menor[es]?\s+que|menos\s+de)\s+(\d+)',
            r'(\d+)\s+dias?\s+ou\s+mais\s+na?\s+uti',
            r'mais\s+de\s+(\d+)\s+dias?\s+uti',
        ]

        for pattern in uti_patterns:
            match = re.search(pattern, question)
            if match:
                groups = match.groups()
                if 'maior' in groups[0] or 'mais' in groups[0]:
                    operator = '>'
                elif 'menor' in groups[0] or 'menos' in groups[0]:
                    operator = '<'
                else:
                    operator = '>='

                value = int(groups[1]) if len(groups) > 1 else int(groups[0])
                condition = QueryCondition('UTI_MES_TO', operator, value)
                query.conditions.append(condition)
                break

    def _extract_groupings(self, question: str, query: SemanticQuery):
        """Extrai cláusulas GROUP BY"""
        # CORRIGIDO: Melhor detecção de agrupamentos baseado em contexto
        grouping_indicators = [
            # Padrões mais específicos primeiro
            ('cidade com', 'CIDADE_RESIDENCIA_PACIENTE'),
            ('cidades com', 'CIDADE_RESIDENCIA_PACIENTE'),
            ('por cidade', 'CIDADE_RESIDENCIA_PACIENTE'),
            ('cada cidade', 'CIDADE_RESIDENCIA_PACIENTE'),
            ('qual cidade', 'CIDADE_RESIDENCIA_PACIENTE'),
            ('quais cidades', 'CIDADE_RESIDENCIA_PACIENTE'),

            ('estado com', 'UF_RESIDENCIA_PACIENTE'),
            ('estados com', 'UF_RESIDENCIA_PACIENTE'),
            ('por estado', 'UF_RESIDENCIA_PACIENTE'),
            ('cada estado', 'UF_RESIDENCIA_PACIENTE'),
            ('qual estado', 'UF_RESIDENCIA_PACIENTE'),
            ('quais estados', 'UF_RESIDENCIA_PACIENTE'),

            ('por sexo', 'SEXO'),
            ('por gênero', 'SEXO'),
            ('cada sexo', 'SEXO'),

            ('por diagnóstico', 'DIAG_PRINC'),
            ('cada diagnóstico', 'DIAG_PRINC'),
            ('diagnóstico com', 'DIAG_PRINC'),
            ('diagnósticos com', 'DIAG_PRINC'),

            ('por faixa etária',
             'CASE WHEN IDADE < 18 THEN "Menor de 18" WHEN IDADE BETWEEN 18 AND 60 THEN "18-60 anos" ELSE "Mais de 60" END'),
            ('por ano', 'SUBSTR(DT_INTER, 1, 4)'),
            ('cada ano', 'SUBSTR(DT_INTER, 1, 4)'),
        ]

        for indicator, column in grouping_indicators:
            if indicator in question:
                query.group_by_columns.append(column)

                # Se detectou agrupamento e não tem agregação específica, usar COUNT
                if not query.aggregations:
                    query.aggregations['*'] = 'COUNT'

                # Se tem palavras de ranking, configurar ordenação e limite
                ranking_words = ['maior', 'menor', 'mais', 'menos', 'principal', 'principais', 'top']
                if any(word in question for word in ranking_words):
                    if not query.limit:
                        query.limit = 10  # Padrão para rankings

                    # Determinar direção da ordenação
                    if any(word in question for word in ['maior', 'mais', 'principal', 'principais', 'top']):
                        direction = "DESC"
                    else:
                        direction = "ASC"

                    # Configurar ordenação baseada na agregação
                    agg_column = list(query.aggregations.keys())[0]
                    agg_type = list(query.aggregations.values())[0]
                    if agg_column == '*' and agg_type == 'COUNT':
                        query.order_by = ("COUNT(*)", direction)
                    else:
                        query.order_by = (f"{agg_type}({agg_column})", direction)

                break

    def _extract_ordering_and_limits(self, question: str, query: SemanticQuery):
        """Extrai ORDER BY e LIMIT"""
        # Identificar limites
        limit_patterns = [
            r'top\s+(\d+)',
            r'(\d+)\s+principais',
            r'(\d+)\s+maiores',
            r'(\d+)\s+primeiros',
            r'primeiros?\s+(\d+)',
            r'principais?\s+(\d+)',
        ]

        for pattern in limit_patterns:
            match = re.search(pattern, question)
            if match:
                query.limit = int(match.group(1))
                # Se tem limite, provavelmente quer ordenação DESC
                if query.aggregations:
                    agg_column = list(query.aggregations.keys())[0]
                    query.order_by = (f"{query.aggregations[agg_column]}({agg_column})", "DESC")
                break

        # Ordenação explícita
        if 'crescente' in question or 'menor para maior' in question:
            if query.aggregations:
                agg_column = list(query.aggregations.keys())[0]
                query.order_by = (f"{query.aggregations[agg_column]}({agg_column})", "ASC")
        elif 'decrescente' in question or 'maior para menor' in question:
            if query.aggregations:
                agg_column = list(query.aggregations.keys())[0]
                query.order_by = (f"{query.aggregations[agg_column]}({agg_column})", "DESC")

    def _add_response_context(self, question: str, query: SemanticQuery):
        """Adiciona contexto para melhorar a resposta"""
        query.response_context.update({
            'original_question': question,
            'has_age_filter': any(cond.column == 'IDADE' for cond in query.conditions),
            'has_geographic_filter': any(
                cond.column in ['CIDADE_RESIDENCIA_PACIENTE', 'UF_RESIDENCIA_PACIENTE'] for cond in query.conditions),
            'is_demographic_analysis': 'SEXO' in query.group_by_columns or any(
                cond.column == 'SEXO' for cond in query.conditions),
            'is_temporal_analysis': any('DT_' in col for col in query.group_by_columns),
        })


class FlexibleSQLGenerator:
    """Gerador de SQL baseado em análise semântica"""

    def __init__(self, table_name: str):
        self.table_name = table_name

    def generate_sql(self, semantic_query: SemanticQuery) -> str:
        """Gera SQL a partir de SemanticQuery"""
        sql_parts = []

        # SELECT clause
        select_clause = self._build_select_clause(semantic_query)
        sql_parts.append(f"SELECT {select_clause}")

        # FROM clause
        sql_parts.append(f"FROM {self.table_name}")

        # WHERE clause
        where_clause = self._build_where_clause(semantic_query)
        if where_clause:
            sql_parts.append(f"WHERE {where_clause}")

        # GROUP BY clause
        if semantic_query.group_by_columns:
            group_by = ", ".join(semantic_query.group_by_columns)
            sql_parts.append(f"GROUP BY {group_by}")

        # ORDER BY clause
        if semantic_query.order_by:
            column, direction = semantic_query.order_by
            sql_parts.append(f"ORDER BY {column} {direction}")

        # LIMIT clause
        if semantic_query.limit:
            sql_parts.append(f"LIMIT {semantic_query.limit}")

        return " ".join(sql_parts)

    def _build_select_clause(self, query: SemanticQuery) -> str:
        """Constrói cláusula SELECT"""
        select_items = []

        # CORRIGIDO: Tratamento especial para contagem com agrupamento
        for column, agg_type in query.aggregations.items():
            if column == '*' and agg_type == 'COUNT':
                select_items.append("COUNT(*)")
            elif '(' in agg_type:  # já formatado (ex: COUNT(DISTINCT col))
                select_items.append(agg_type)
            else:
                select_items.append(f"{agg_type}({column})")

        # IMPORTANTE: Colunas de agrupamento devem vir ANTES das agregações no SELECT
        select_items = query.group_by_columns + select_items

        # Colunas alvo (apenas se não há agrupamento)
        if not query.group_by_columns:
            select_items.extend(query.target_columns)

        # Se não há nada específico, usar COUNT(*)
        if not select_items:
            select_items.append("COUNT(*)")

        return ", ".join(select_items)

    def _build_where_clause(self, query: SemanticQuery) -> str:
        """Constrói cláusula WHERE"""
        if not query.conditions:
            return ""

        where_parts = []
        for i, condition in enumerate(query.conditions):
            if i > 0:
                where_parts.append(condition.logical_connector)
            where_parts.append(condition.to_sql())

        return " ".join(where_parts)


class EnhancedAdvancedSQLInterface:
    """Interface SQL avançada com análise semântica"""

    def __init__(self, db_uri, table_name):
        self.db = SQLDatabase.from_uri(db_uri, include_tables=[table_name])
        self.table_name = table_name
        self.llm = None
        self.schema_cache = None
        self.data_profile = None
        self.semantic_analyzer = SemanticAnalyzer()
        self.sql_generator = FlexibleSQLGenerator(table_name)
        self.conversation_context = []

    def load_llm(self):
        """Carrega o LLM se disponível"""
        try:
            self.llm = carregar_llm_langchain(use_gpu_if_available=True)
            return self.llm is not None
        except Exception as e:
            print(f"Falha ao carregar LLM: {e}")
            return False

    def get_table_info(self):
        """Retorna informações sobre a tabela com cache"""
        if not self.schema_cache:
            self.schema_cache = self.db.get_table_info([self.table_name])
        return self.schema_cache

    def profile_data(self):
        """Cria um perfil abrangente dos dados"""
        if self.data_profile:
            return self.data_profile

        try:
            basic_stats = self._get_basic_statistics()
            column_analysis = self._analyze_columns()
            distributions = self._get_key_distributions()

            self.data_profile = {
                'basic_stats': basic_stats,
                'columns': column_analysis,
                'distributions': distributions,
                'last_updated': datetime.now().isoformat()
            }

            return self.data_profile

        except Exception as e:
            print(f"Erro ao criar perfil: {e}")
            return self._get_minimal_profile()

    def _get_basic_statistics(self) -> Dict:
        """Obtém estatísticas básicas da tabela"""
        try:
            total_result = self.execute_sql(f"SELECT COUNT(*) FROM {self.table_name}")
            total_records = self._parse_single_value(total_result)

            age_stats = self.execute_sql(f"SELECT MIN(IDADE), MAX(IDADE), AVG(IDADE) FROM {self.table_name}")
            age_data = self._parse_tuple_result(age_stats)

            return {
                'total_records': total_records,
                'age_range': {
                    'min': int(age_data[0]) if age_data else 0,
                    'max': int(age_data[1]) if age_data else 0,
                    'avg': round(age_data[2], 1) if age_data else 0
                }
            }
        except:
            return {'total_records': 0, 'age_range': {'min': 0, 'max': 0, 'avg': 0}}

    def _analyze_columns(self) -> Dict:
        """Analisa as colunas da tabela"""
        return {
            'DIAG_PRINC': {'type': 'categorical', 'description': 'Diagnóstico Principal (CID)'},
            'IDADE': {'type': 'numeric', 'description': 'Idade do paciente'},
            'SEXO': {'type': 'categorical', 'description': 'Sexo (1=Masculino, 3=Feminino)'},
            'CIDADE_RESIDENCIA_PACIENTE': {'type': 'categorical', 'description': 'Cidade de residência'},
            'UF_RESIDENCIA_PACIENTE': {'type': 'categorical', 'description': 'Estado de residência'},
            'VAL_TOT': {'type': 'numeric', 'description': 'Valor total do procedimento'},
            'MORTE': {'type': 'binary', 'description': 'Indicador de óbito (0=Não, 1=Sim)'},
            'UTI_MES_TO': {'type': 'numeric', 'description': 'Dias em UTI'},
            'DT_INTER': {'type': 'date', 'description': 'Data de internação'},
            'DT_SAIDA': {'type': 'date', 'description': 'Data de saída'}
        }

    def _get_key_distributions(self) -> Dict:
        """Obtém distribuições chave dos dados"""
        try:
            distributions = {}

            cities_result = self.execute_sql(f"""
                SELECT CIDADE_RESIDENCIA_PACIENTE, COUNT(*) 
                FROM {self.table_name} 
                GROUP BY CIDADE_RESIDENCIA_PACIENTE 
                ORDER BY COUNT(*) DESC LIMIT 5
            """)
            distributions['top_cities'] = self._parse_list_result(cities_result)

            gender_result = self.execute_sql(f"""
                SELECT SEXO, COUNT(*) 
                FROM {self.table_name} 
                GROUP BY SEXO
            """)
            distributions['gender'] = self._parse_list_result(gender_result)

            return distributions
        except:
            return {}

    def execute_sql(self, query):
        """Executa uma query SQL com tratamento de erro melhorado"""
        try:
            result = self.db.run(query)
            return result
        except Exception as e:
            return f"Erro na query: {str(e)[:200]}..."

    def _parse_single_value(self, result):
        """Parse de resultado com valor único"""
        try:
            if isinstance(result, str):
                clean_result = result.strip('[]()').split(',')[0]
                return int(float(clean_result))
            return result
        except:
            return 0

    def _parse_tuple_result(self, result):
        """Parse de resultado tupla"""
        try:
            if isinstance(result, str) and '(' in result:
                stats_clean = result.strip('[]()').split(',')
                return [float(x.strip()) for x in stats_clean]
            return None
        except:
            return None

    def _parse_list_result(self, result):
        """Parse de resultado lista - melhorado"""
        try:
            if isinstance(result, str):
                # Implementar parse mais robusto para resultados complexos
                if result.startswith('[') and result.endswith(']'):
                    # Remove colchetes externos
                    content = result[1:-1]
                    # Parse de tuplas
                    items = []
                    if '(' in content:
                        # Resultado com tuplas
                        tuples = re.findall(r'\([^)]+\)', content)
                        for tuple_str in tuples:
                            # Remove parênteses e divide por vírgula
                            values = tuple_str.strip('()').split(',')
                            parsed_values = []
                            for val in values:
                                val = val.strip().strip("'\"")
                                try:
                                    # Tenta converter para número
                                    parsed_values.append(int(val))
                                except:
                                    # Mantém como string
                                    parsed_values.append(val)
                            items.append(tuple(parsed_values))
                        return items
                return []
            return result
        except:
            return []

    def answer_question(self, question: str) -> str:
        """Responde pergunta usando análise semântica avançada"""
        # Adicionar ao contexto da conversa
        self.conversation_context.append({
            'question': question,
            'timestamp': datetime.now().isoformat()
        })

        try:
            # 1. Analise semantica da pergunta
            semantic_query = self.semantic_analyzer.analyze_question(question)

            # 2. Validar se conseguiu extrair informacoes suficientes
            if not semantic_query.aggregations and not semantic_query.target_columns and not semantic_query.conditions:
                return self._handle_unrecognized_question(question)

            # 3. Gerar SQL a partir da analise semantica
            sql_query = self.sql_generator.generate_sql(semantic_query)

            # 4. Executar consulta
            result = self.execute_sql(sql_query)

            if isinstance(result, str) and result.startswith("Erro"):
                # Se falhou, tentar com LLM como fallback
                if self.llm:
                    return self._generate_llm_response(question)
                else:
                    return f"ERRO: {result}"

            # 5. Formatar resposta contextualizada
            formatted_response = self._format_semantic_response(semantic_query, result, question)

            # 6. Adicionar query executada para transparencia
            formatted_response += f"\n\nQuery executada: {sql_query}"

            return formatted_response

        except Exception as e:
            # Fallback para LLM se análise semântica falhar
            if self.llm:
                return self._generate_llm_response(question)
            else:
                return f"ERRO ao processar pergunta: {str(e)[:200]}..."

    def _format_semantic_response(self, semantic_query: SemanticQuery, result: Any, question: str) -> str:
        """Formata resposta baseada na análise semântica"""
        try:
            # Parse do resultado
            parsed_result = self._parse_single_value(result)

            # Determiner tipo de resposta baseado na consulta
            if semantic_query.aggregations:
                agg_column = list(semantic_query.aggregations.keys())[0]
                agg_type = list(semantic_query.aggregations.values())[0]

                # Resposta contextualizada para diferentes tipos de agregação
                if 'AVG' in agg_type and agg_column == 'UTI_MES_TO':
                    response = f"Media de dias em UTI"

                    # Adicionar contexto das condições
                    conditions_text = self._format_conditions_context(semantic_query.conditions)
                    if conditions_text:
                        response += f" {conditions_text}"

                    response += f": {parsed_result:.1f} dias"

                    # Adicionar comparação com média geral se houver filtros
                    if semantic_query.conditions:
                        try:
                            general_avg_result = self.execute_sql(
                                f"SELECT AVG(UTI_MES_TO) FROM {self.table_name} WHERE UTI_MES_TO > 0")
                            general_avg = self._parse_single_value(general_avg_result)
                            if general_avg != parsed_result:
                                diff = parsed_result - general_avg
                                direction = "maior" if diff > 0 else "menor"
                                response += f"\nComparacao: {abs(diff):.1f} dias {direction} que a media geral ({general_avg:.1f} dias)"
                        except:
                            pass

                elif 'AVG' in agg_type and agg_column == 'IDADE':
                    response = f"Idade media"
                    conditions_text = self._format_conditions_context(semantic_query.conditions)
                    if conditions_text:
                        response += f" {conditions_text}"
                    response += f": {parsed_result:.1f} anos"

                elif 'AVG' in agg_type and agg_column == 'VAL_TOT':
                    response = f"Valor medio"
                    conditions_text = self._format_conditions_context(semantic_query.conditions)
                    if conditions_text:
                        response += f" {conditions_text}"
                    response += f": R$ {parsed_result:,.2f}"

                elif 'COUNT' in agg_type:
                    # CORRIGIDO: Melhor formatação para resultados com agrupamento
                    if semantic_query.group_by_columns:
                        # Se tem agrupamento, mostrar como ranking/lista
                        group_column = semantic_query.group_by_columns[0]

                        # Determinar o que está sendo agrupado
                        if 'CIDADE' in group_column:
                            response = f"Cidades"
                        elif 'UF' in group_column:
                            response = f"Estados"
                        elif 'SEXO' in group_column:
                            response = f"Distribuicao por sexo"
                        elif 'DIAG' in group_column:
                            response = f"Diagnosticos"
                        else:
                            response = f"Agrupamento"

                        # Adicionar contexto das condições
                        conditions_text = self._format_conditions_context(semantic_query.conditions)
                        if conditions_text:
                            response += f" {conditions_text}"

                        response += f":\n{self._format_grouped_result(result, semantic_query)}"
                    else:
                        # Contagem simples
                        response = f"Total de registros"
                        conditions_text = self._format_conditions_context(semantic_query.conditions)
                        if conditions_text:
                            response += f" {conditions_text}"
                        response += f": {parsed_result:,}"

                        # Calcular percentual se houver filtros
                        if semantic_query.conditions:
                            try:
                                total_result = self.execute_sql(f"SELECT COUNT(*) FROM {self.table_name}")
                                total_records = self._parse_single_value(total_result)
                                percentage = (parsed_result / total_records) * 100 if total_records > 0 else 0
                                response += f" ({percentage:.1f}% do total)"
                            except:
                                pass

                elif 'MAX' in agg_type:
                    response = f"Valor maximo"
                    conditions_text = self._format_conditions_context(semantic_query.conditions)
                    if conditions_text:
                        response += f" {conditions_text}"

                    if agg_column == 'IDADE':
                        response += f": {parsed_result} anos"
                    elif agg_column == 'VAL_TOT':
                        response += f": R$ {parsed_result:,.2f}"
                    else:
                        response += f": {parsed_result}"

                elif 'MIN' in agg_type:
                    response = f"Valor minimo"
                    conditions_text = self._format_conditions_context(semantic_query.conditions)
                    if conditions_text:
                        response += f" {conditions_text}"

                    if agg_column == 'IDADE':
                        response += f": {parsed_result} anos"
                    elif agg_column == 'VAL_TOT':
                        response += f": R$ {parsed_result:,.2f}"
                    else:
                        response += f": {parsed_result}"

                else:
                    response = f"Resultado: {parsed_result}"

                return response

            else:
                # Resposta para consultas não agregadas
                return f"Resultado da consulta: {result}"

        except Exception as e:
            return f"Resultado: {result}"

    def _format_conditions_context(self, conditions: List[QueryCondition]) -> str:
        """Formata as condições para exibição amigável"""
        if not conditions:
            return ""

        context_parts = []
        for condition in conditions:
            if condition.column == 'IDADE':
                if condition.operator == '>':
                    context_parts.append(f"para idades maiores que {condition.value} anos")
                elif condition.operator == '<':
                    context_parts.append(f"para idades menores que {condition.value} anos")
                elif condition.operator == '>=':
                    context_parts.append(f"para idades de {condition.value} anos ou mais")
                elif condition.operator == '<=':
                    context_parts.append(f"para idades ate {condition.value} anos")
                elif condition.operator == '=':
                    context_parts.append(f"para idade de {condition.value} anos")
                elif 'BETWEEN' in condition.operator:
                    context_parts.append(f"para idades entre {condition.value}")

            elif condition.column == 'SEXO':
                if condition.value == 1:
                    context_parts.append("para pacientes masculinos")
                elif condition.value == 3:
                    context_parts.append("para pacientes femininos")

            elif condition.column == 'CIDADE_RESIDENCIA_PACIENTE':
                city_name = condition.value.replace('%', '').strip("'")
                context_parts.append(f"em {city_name}")

            elif condition.column == 'MORTE':
                if condition.value == 1:
                    context_parts.append("com obito")
                elif condition.value == 0:
                    context_parts.append("sem obito")

            elif condition.column == 'UTI_MES_TO':
                if condition.operator == '>':
                    context_parts.append(f"com mais de {condition.value} dias em UTI")
                elif condition.operator == '<':
                    context_parts.append(f"com menos de {condition.value} dias em UTI")
                elif condition.operator == '>=':
                    context_parts.append(f"com pelo menos {condition.value} dias em UTI")

        return " ".join(context_parts) if context_parts else ""

    def _format_grouped_result(self, result: Any, semantic_query: SemanticQuery) -> str:
        """Formata resultado de consultas com agrupamento"""
        try:
            # Parse do resultado para lista de tuplas
            if isinstance(result, str):
                # Tentar fazer parse do resultado string
                parsed_results = self._parse_list_result(result)
                if not parsed_results:
                    return str(result)
            else:
                parsed_results = result

            # Se é lista de tuplas, formatar adequadamente
            if isinstance(parsed_results, list) and len(parsed_results) > 0:
                formatted_lines = []
                for i, item in enumerate(parsed_results[:10], 1):  # Mostrar top 10
                    if isinstance(item, tuple) and len(item) == 2:
                        name, count = item
                        formatted_lines.append(f"{i}. {name}: {count:,}")
                    else:
                        formatted_lines.append(f"{i}. {item}")

                return "\n".join(formatted_lines)
            else:
                return str(parsed_results)

        except Exception as e:
            return str(result)

    def _add_query_metadata(self, sql_query: str, semantic_query: SemanticQuery) -> str:
        """Adiciona metadados sobre a consulta"""
        metadata = []

        # Número de condições aplicadas
        if semantic_query.conditions:
            metadata.append(f"Filtros aplicados: {len(semantic_query.conditions)}")

        # Tipo de análise
        analysis_type = []
        if semantic_query.response_context.get('has_age_filter'):
            analysis_type.append("demografica")
        if semantic_query.response_context.get('has_geographic_filter'):
            analysis_type.append("geografica")
        if semantic_query.response_context.get('is_temporal_analysis'):
            analysis_type.append("temporal")

        if analysis_type:
            metadata.append(f"Tipo de analise: {', '.join(analysis_type)}")

        # Query SQL executada
        metadata.append(f"Query executada: {sql_query}")

        return "\n\n" + "\n".join(metadata)

    def _handle_unrecognized_question(self, question: str) -> str:
        """Trata perguntas não reconhecidas pela análise semântica"""
        if self.llm:
            return self._generate_llm_response(question)
        else:
            return self._suggest_semantic_alternatives(question)

    def _generate_llm_response(self, question: str) -> str:
        """Gera resposta usando LLM como fallback"""
        try:
            schema = self.get_table_info()
            profile = self.profile_data()

            context = f"""
            Voce e um especialista em analise de dados do SUS.
            
            Schema da tabela:
            {schema}
            
            Perfil dos dados:
            - Total de registros: {profile.get('basic_stats', {}).get('total_records', 0):,}
            - Faixa etaria: {profile.get('basic_stats', {}).get('age_range', {})}
            
            Pergunta: {question}
            
            Gere uma query SQL que responda a pergunta. Retorne APENAS a query SQL.
            """

            response = self.llm.invoke(context)

            # Extrair SQL
            sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql = sql_match.group(1).strip()
            else:
                lines = response.strip().split('\n')
                sql = None
                for line in lines:
                    if line.strip().upper().startswith('SELECT'):
                        sql = line.strip()
                        break

                if not sql:
                    return "Nao consegui gerar uma consulta apropriada para sua pergunta."

            # Executar query gerada
            result = self.execute_sql(sql)

            if result.startswith("Erro"):
                return f"Query gerada pelo LLM falhou: {result}"

            return f"Resposta via LLM:\n{result}\n\nQuery: {sql}"

        except Exception as e:
            return f"Erro no LLM: {str(e)[:200]}..."

    def _suggest_semantic_alternatives(self, question: str) -> str:
        """Sugere alternativas baseadas na análise semântica"""
        return f"""
        Nao consegui interpretar completamente sua pergunta usando analise semantica.
        
        Para perguntas com filtros, use estruturas como:
        - "media de [campo] para [condicao]"
        - "total de casos com [condicao]"
        - "valor maximo para [filtro]"
        
        Exemplos especificos:
        - "media de dias na UTI para idades maiores que 60 anos"
        - "total de casos para pacientes femininos"
        - "valor medio para casos em Porto Alegre"
        - "idade media para pacientes com mais de 5 dias em UTI"
        
        Estruturas suportadas:
        - Condicoes de idade: "maior que X anos", "entre X e Y anos", "menores que X"
        - Filtros geograficos: "em [cidade]", "no estado de [UF]"
        - Condicoes medicas: "com obito", "sem obito", "com mais de X dias em UTI"
        - Filtros demograficos: "pacientes masculinos", "pacientes femininos"
        
        Dica: Seja especifico sobre o que quer calcular e quais filtros aplicar.
        """

    def _get_minimal_profile(self) -> Dict:
        """Retorna perfil mínimo em caso de erro"""
        return {
            'basic_stats': {'total_records': 0, 'age_range': {'min': 0, 'max': 100, 'avg': 50}},
            'columns': {},
            'distributions': {}
        }

    def get_conversation_summary(self) -> str:
        """Retorna resumo da conversa atual"""
        if not self.conversation_context:
            return "Nenhuma pergunta feita ainda."

        return f"Resumo da sessao: {len(self.conversation_context)} perguntas realizadas"

    def debug_semantic_analysis(self, question: str) -> str:
        """Funcao de debug para mostrar analise semantica"""
        semantic_query = self.semantic_analyzer.analyze_question(question)

        debug_info = f"""
            Debug da Analise Semantica:
            
            Pergunta: {question}
            
            Agregacoes detectadas: {semantic_query.aggregations}
            Colunas alvo: {semantic_query.target_columns}
            Condicoes WHERE: {[f"{cond.column} {cond.operator} {cond.value}" for cond in semantic_query.conditions]}
            Agrupamentos: {semantic_query.group_by_columns}
            Ordenacao: {semantic_query.order_by}
            Limite: {semantic_query.limit}
            
            SQL Gerado: 
            {self.sql_generator.generate_sql(semantic_query)}
            """
        return debug_info

    def test_common_queries(self):
        """Testa consultas comuns para verificar funcionamento"""
        test_queries = [
            "total de casos para mulheres com mais de 60 anos",
            "media de dias na UTI para idades maiores que 60 anos",
            "total de casos para mulheres",
            "quantos registros temos",
            "idade media dos pacientes"
        ]

        print("\n=== TESTE DE CONSULTAS COMUNS ===")
        for query in test_queries:
            print(f"\nTestando: '{query}'")
            try:
                semantic_query = self.semantic_analyzer.analyze_question(query)
                sql_generated = self.sql_generator.generate_sql(semantic_query)
                print(f"SQL: {sql_generated}")
                print(f"Agregacoes: {semantic_query.aggregations}")
                print(f"Condicoes: {[(c.column, c.operator, c.value) for c in semantic_query.conditions]}")
            except Exception as e:
                print(f"ERRO: {e}")
        print("=== FIM DOS TESTES ===\n")


def main():
    print("=== Interface SQL Semantica para Dados do SUS ===")
    print("POC - Transformacao de Pergunta em SQL")

    # Preparar banco de dados
    print(f"\n[ETAPA 1/2] Preparando banco de dados...")
    if not os.path.exists(CSV_FILE_PATH):
        print(f"ERRO: Arquivo CSV '{CSV_FILE_PATH}' nao encontrado.")
        return

    try:
        criar_banco_de_dados_do_csv(
            csv_path=CSV_FILE_PATH,
            db_name=DB_FILE_NAME,
            table_name_override=TABLE_NAME
        )
        print(f"Banco de dados pronto: {DB_FILE_NAME}")
    except Exception as e:
        print(f"ERRO ao criar banco: {e}")
        return

    # Configurar interface
    print(f"\n[ETAPA 2/2] Configurando interface...")
    try:
        interface = EnhancedAdvancedSQLInterface(DB_URI, TABLE_NAME)
        print("Interface SQL configurada.")

        # Carregar perfil dos dados
        profile = interface.profile_data()
        print(f"Dados carregados: {profile.get('basic_stats', {}).get('total_records', 0):,} registros")

        # Tentar carregar LLM
        if interface.load_llm():
            print("LLM disponivel como fallback.")
        else:
            print("LLM nao disponivel - usando apenas analise semantica.")

    except Exception as e:
        print(f"ERRO ao configurar interface: {e}")
        return

    # Interface interativa simples
    print(f"\nInterface ativa! Digite suas perguntas sobre os dados do SUS.")
    print("Comandos: 'debug: pergunta' | 'teste' | 'sql: query' | 'sair'")
    print("=" * 60)

    while True:
        try:
            pergunta = input("\nSua pergunta: ").strip()

            if pergunta.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando. Obrigado!")
                break

            if not pergunta:
                continue

            # Comandos especiais
            if pergunta.lower().startswith('debug:'):
                debug_question = pergunta[6:].strip()
                debug_info = interface.debug_semantic_analysis(debug_question)
                print(f"\n{debug_info}")
                continue

            if pergunta.lower() == 'teste':
                interface.test_common_queries()
                continue

            # Comando SQL direto
            if pergunta.lower().startswith('sql:'):
                sql_query = pergunta[4:].strip()
                result = interface.execute_sql(sql_query)
                print(f"\nResultado: {result}")
            else:
                # Analise semantica e resposta
                resposta = interface.answer_question(pergunta)
                print(f"\n{resposta}")

        except KeyboardInterrupt:
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"Erro: {e}")


if __name__ == "__main__":
    main()