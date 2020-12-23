import xml.etree.ElementTree as et
import time
import json

# data1 = et.parse('D:/BCDD/Documents/Tal/Projects/HMDB/DataSets/saliva_metabolites/saliva_metabolites.xml')
# data1 = et.parse('D:/BCDD/Documents/Tal/Projects/HMDB/DataSets/serum_metabolites/serum_metabolites.xml')

def parser_hmdb(pathtoxmlfile, pathtojsonfile):
    '''
    The function receive an XML file from HMDB and parse it while selecting only few nodes.
    :param pathtoxmlfile: Is a path to a XML input file in local library from the HMDB.
    :param pathtojsonfile: Is a path to a JSON output file in local library .
    :return: A list of dictionaries. Each dictionary represents a metabolite with 12 keys
    '''

    data = et.parse(pathtoxmlfile)

    root = data.getroot()
    # name space
    ns = {"h": "http://www.hmdb.ca"}

    # extract the first 3 metabolites
    metabolites = root.findall('./h:metabolite', ns) #[0:3]

    # loop over all nodes and select only few in order to create a list of dictionaries
    newlist = []
    start_time = time.time()
    for child in metabolites:
        innerlistsyn = []
        innerlistpath=[]
        innerlistdis=[]
        dicts = {}
        for subchild in child:
            # if the node tag is "accession" the create a key with called "accession" with the value in that node
            if subchild.tag == '{http://www.hmdb.ca}accession':
                dicts["accession"] = subchild.text
            if subchild.tag == '{http://www.hmdb.ca}secondary_accessions':
                dicts["second_accession"] = []
                for second in subchild:
                    dicts["second_accession"].append(second.text)
            if subchild.tag == '{http://www.hmdb.ca}name':
                dicts["name"] = subchild.text
                # innerlist.append(subchild.text)
            if subchild.tag == '{http://www.hmdb.ca}description':
                dicts["description"] = subchild.text

            # if the node tag is "synonyms" the create a key with called "synonyms" with a list of values in that node
            if subchild.tag == '{http://www.hmdb.ca}synonyms':
                for synonym in subchild:
                    innerlistsyn.append(synonym.text)
                    # print(innerlist)
                dicts["synonyms"] = innerlistsyn

            if subchild.tag == '{http://www.hmdb.ca}chemical_formula':
                dicts["chemical_formula"] = subchild.text
            if subchild.tag == '{http://www.hmdb.ca}smiles':
                dicts["smiles"] = subchild.text
            if subchild.tag == '{http://www.hmdb.ca}inchikey':
                dicts["inchikey"] = subchild.text

            # similr to the "synonyms" create a list of values to the key "pathway_name"
            if subchild.tag == '{http://www.hmdb.ca}biological_properties':
                for pathways in subchild:
                    if pathways.tag == '{http://www.hmdb.ca}pathways':
                        for pathway in pathways:
                            if pathway.tag == '{http://www.hmdb.ca}pathway':
                                for name in pathway:
                                    if name.tag == '{http://www.hmdb.ca}name':
                                        innerlistpath.append(name.text)
                                # print(innerlist)
                dicts["pathway_name"]= innerlistpath

            if subchild.tag == '{http://www.hmdb.ca}diseases':
                for disease in subchild:
                    if disease.tag == '{http://www.hmdb.ca}disease':
                        for name in disease:
                            if name.tag == '{http://www.hmdb.ca}name':
                                innerlistdis.append(name.text)
                dicts["disease_name"] = innerlistdis
            # the first variation for extracting disease and pumedbid+ refer_ text is creating for each disease a
            # variables / dictionaries   called "disease_name pumed_id" and   "disease_name reference_text"
            # for each disease in the "disease_name" key create 2 dictionary value the first,
            # a list of all pubmed Id's to that disease and the second a list of all titles of the pubmed Id's.
            # if subchild.tag == '{http://www.hmdb.ca}diseases':
            #     for disease in subchild:
            #         if disease.tag == '{http://www.hmdb.ca}disease':
            #             for childisease in disease:  # childisease = is  references
            #                 if childisease.tag == '{http://www.hmdb.ca}name':
            #                     diseasekey=childisease.text
            #                     innerlistpubmed_id = []
            #                     innerlistrefe_text = []
            #                     # innerlistdisname.append(diseasekey.text)
            #                     # print(diseasekey)
            #                 if childisease.tag== '{http://www.hmdb.ca}references':
            #                     for reference in childisease:
            #                         if reference.tag == '{http://www.hmdb.ca}reference':
            #                             # print(reference.tag)
            #                             for childref in reference:
            #                                 # print (pubmed_id.tag)
            #                                 if childref.tag == '{http://www.hmdb.ca}pubmed_id':
            #                                     # print(pubmed_id.text)
            #                                     innerlistpubmed_id.append(childref.text)
            #                                 if childref.tag == '{http://www.hmdb.ca}reference_text':
            #                                     # print(pubmed_id.text)
            #                                     innerlistrefe_text.append(childref.text)
            #                     dicts[diseasekey+' pumed_id'] = innerlistpubmed_id
            #                     dicts[diseasekey+' reference_text'] = innerlistrefe_text
            # The second way creating for each disease dictionary another 2 dictionaries with pubmd_id and  refe_text
            # for each disease in the "disease_name" key create 2 dictionary value the first,
            # a list of all pubmed Id's to that disease and the second a list of all titles of the pubmed Id's.
            if subchild.tag == '{http://www.hmdb.ca}diseases':
                for disease in subchild:
                    if disease.tag == '{http://www.hmdb.ca}disease':
                        for childisease in disease:  # childisease = is  references
                            if childisease.tag == '{http://www.hmdb.ca}name':
                                diseasekey = childisease.text
                                innerlistpubmed_id = []
                                innerlistrefe_text = []
                                innerdictpubmed_id = {}
                                innerdictrefe_text = {}
                                # innerlistdisname.append(diseasekey.text)
                                # print(diseasekey)
                            if childisease.tag == '{http://www.hmdb.ca}references':
                                for reference in childisease:
                                    if reference.tag == '{http://www.hmdb.ca}reference':
                                        # print(reference.tag)
                                        for childref in reference:
                                            # print (pubmed_id.tag)
                                            if childref.tag == '{http://www.hmdb.ca}pubmed_id':
                                                # print(pubmed_id.text)
                                                innerlistpubmed_id.append(childref.text)
                                            if childref.tag == '{http://www.hmdb.ca}reference_text':
                                                # print(pubmed_id.text)
                                                innerlistrefe_text.append(childref.text)
                                                innerdictrefe_text["refe_text"] = innerlistrefe_text
                                                innerdictpubmed_id["pubmed_id"] = innerlistpubmed_id
                                dicts[diseasekey] = {}
                                dicts[diseasekey]["pubmed_id"] = innerlistpubmed_id
                                dicts[diseasekey]["refe_text"] = innerlistrefe_text
                newdict = dicts

            if subchild.tag == '{http://www.hmdb.ca}kegg_id':
                dicts["kegg_id"] = subchild.text
            if subchild.tag == '{http://www.hmdb.ca}meta_cyc_id':
                dicts["meta_cyc_id"] = subchild.text

        newlist.append(dicts)

    print("--- %s seconds ---" % (time.time() - start_time))

    JSON_metabolites = newlist
    # create a  JSON file in with the parse data from the HMDB
    with open(pathtojsonfile, 'w') as fout:
        json.dump(JSON_metabolites, fout, indent=4)

    # return (newlist)

if __name__ == "__main__":

    parser_hmdb('HMDB/urine_metabolites.xml',
                'HMDB/urine_metabolites.json')

    parser_hmdb('HMDB/serum_metabolites.xml',
                'HMDB/serum_metabolites.json')