# -*- coding:utf-8 -*-
# @Time    :2023/10/9 17:14
# @Author  :ZZK
# @ File   :10-9.py
# Description:
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from langdetect import detect
# 输入名词实体
# entity = "china"  # 替换为您要查询的名词实体
def sparqlw(entity):
    # 1. 查询Wikidata以获取实体的QID（唯一标识符）
    wikidata_endpoint = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": entity
    }
    response = requests.get(wikidata_endpoint, params=params)
    data = response.json()
    try:
        qid = data["search"][0]["id"]  # 获取QID
    except:
        return ""

    # 2. 使用QID执行SPARQL查询以获取与实体相关的三元组
    sparql_endpoint = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?property ?propertyLabel ?entity2 ?entity2Label
    WHERE {{
        wd:{qid} ?property ?entity2.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 200
    """
    sparql = SPARQLWrapper(sparql_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
    except:
        return ""

    # 3. 解析和显示查询结果，包括关系URL的自然语言描述
    # 保留补充的10条信息吧。
    n = 0
    entity_k = []
    for result in results["results"]["bindings"]:
        property_uri = result["property"]["value"]  # 获取关系URL
        property_label = result["propertyLabel"]["value"]  # 获取关系的自然语言描述
        entity2_label = result["entity2Label"]["value"]
        try:
            detected_l = detect(entity2_label)
        except:
            detected_l = "None"
        if detected_l=='en' or detected_l=='zh':
            n+=1
            entity_k.append(entity2_label)
            if n>10:
                break
            # print(f"Relation: {property_label} ({property_uri})")
            # print(f"Entity2: {entity2_label}")
            # print("--------")
    return "\t".join(entity_k)