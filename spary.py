# -*- coding:utf-8 -*-
# @Time    :2023/9/9 17:14
# @Author  :ZZK
# @ File   :
# Description:
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from langdetect import detect
def sparqlw(entity):
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

    n = 0
    entity_k = []
    for result in results["results"]["bindings"]:
        property_uri = result["property"]["value"]  
        property_label = result["propertyLabel"]["value"] 
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
