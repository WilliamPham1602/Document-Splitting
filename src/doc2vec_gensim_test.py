from gensim.models import Doc2Vec
import nltk
from bs4 import BeautifulSoup
import re
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

model = Doc2Vec.load("./models/pdf_split_d2v_gensim_1024_db.mod")

model_dbow = Doc2Vec.load("./models/pdf_split_d2v_gensim_{}_db.mod".format(1024))
model_dmm = Doc2Vec.load("./models/pdf_split_d2v_gensim_{}_dm.mod".format(1024))

# Concatenate model
model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

text = ' \n\n \n\nCluster\n\nContactpersoon\nTelefoon\n\nUw brief\n\nOns kenmerk\nBijlage(n)\n\nOnderwerp\n\ngemeente\nHaarlemmermeer\n\nPostbus 250\n2130 AG Hoofddorp\n\nBezoekadres:\nRaadhuisplein 1\nHoofddorp\n\nTelefoon 0900 1852\nTelefax 023 563 95 50\n\nDienstverlening\n\n0900-1852 Hoofddorp. 1 november 2013\nd.d. 22 oktober 2013 Verzenddatum\n|-13.58484\\verg\n\nzie onder 4 NÛV 2ma\n\nOntheffing artikel 5:25 van de Algemene plaatselijke\nverordening Ringvaart-Akerdijk 47 Badhoevedorp.\n\nGeachte\nGeachte\n\nAanvraag\n\nOp 25 oktober 2013 hebben wij uw aanvraag ontvangen om ontheffing voor het innemen van\neen ligplaats met het woonschip "Andromeda" aan de Ringvaart-Akerdijk 47 te\nBadhoevedorp. Het betreft hier een nieuwe ontheffing omdat u uw woonark van 21 meter\nlengte gaat uitbreideh tot 25 m.\n\nDe afmetingen van het woonschip zijn dan - volgens het aanvraagformulier en bijgevoegde\ntekening - 25,00 m lengte x 6,00 m breedte x 4,15 m hoogte -.\n\nJuridisch kader\n\n-Algemene plaatselijke verordening\n\nOp 1 augustus 2013 is de gewijzigde Algemene Plaatselijke Verordening 2012 (APV) in\nwerking getreden.\n\nVan belang is artikel 5:25 van de APV, dat bepaalt dat wij voor een ligplaats voor een\nwoonschip een ontheffing kunnen afgeven. In lid drie van dit artikel zijn de toetsingscriteria\ngegeven waarop een aanvraag geweigerd kan worden, zoals openbare orde,\n(verkeers)veiligheid, volksgezondheid. Deze criteria zijn niet van toepassing zodat er wat dat\nbetreft geen belemmering is om de ontheffing te verlenen.\n\nVoorts zijn in artikel 5:27 1° lid onder b de maximale afmetingen opgenomen. Tevens staat\ndaar dat een ontheffing geweigerd moet worden indien de afstand tussen twee\nwoonschepen op grond van brandveiligheid niet voldoende is.\n\n \n\n \n\n-Bestemmingsplan\n\nTer plaatse is het bestemmingsplan “Badhoevedorp-Lijnden-Oost" in ontwerp vastgesteld.\nDe onderhavige ligplaats is in dit bestemmingsplan opgenomen. \'\n\n \n\n \n'

text2 = "S eurofins\n\nPr\n\nAnalytical Report to Order 01502042\n\nonitoring\n\nAR-15-WE-000743-01 Page 650f82\nProject PMO11083-15-01, K62\nSample No. Customer M10b, 201g\nDurchslag\nLab-D # 01502042064\n[Parameter [Unit |LOG [LOD [UOM _ [Method\nInorganic compounds\nAmmonia (NH3) mg/’samp{0,008 |0,0027 |0,063 [VDi3ss6Pan1 0,00800\nTetal 502 (sulphur diomde) mg/samp{0,01 0,0033 |021 EN 14791 -\nHalogens and compounds\n[Hydrochlorie acid (HCI) Ima'samp{0,02 _ |0,007 |0029 |EN | - |\nAnalytical Report to Order 01502042\nAR-15-WE-000743-01 Page66of82\nProject PMO11083-15-01, K62\nSample No. Customer M11a,213g\nLab-D # 01502042065\n[Parameter [Unit |LOQ |LOD [|[uOM _ [Method\nInorganic compounds\nAmmonia (NH3) mg/samp{0,008 0,0027 1|0,063 VDI 3496 Pan 1 0.0310\nTotal 502 (sulphur diomde) mg/samp{0,01 0,0033 |0,21 EN 14791 7\nHalogens and compounds\n[Hydrochlorie acid (HCI) Img'samp{0,02 _ |0,007 |0029 |EN19n | - |\nAnalytical Report to Order 01502042\nAR-15-WE-000743-01 Pageê7of82\nProject PMO11083-15-01, K62\nSample No. Customer M11b, 214g\nDurchslag\nLab-D # 01502042066\n[Parameter [Unit |[LOG [LOD [UOM [Method\nInorganic compounds\nAmmonia (NH3) mg/samp{0,008 0,0027 |0,063 [VDI 3436 Part 1 0.0130\nTotal 502 (sulphur diomde ) mg/samp40,01 0,0033 |021 EN 14791 7\nHalogens and compounds\n[Hydrochloric acid (HCI) Img'samp{0,02 |0007 |0029 _ |EN19n\n\nr011083-15-01 QAL2\n\npagina 59 van 113\n"
import pandas as pd

# df = pd.read_csv('./data/0.0_0519.csv')



def cleanText(text):
    text = BeautifulSoup(text, "html.parser").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'\\n', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

text = cleanText(text)
text_vec = tokenize_text(text)

text2 = cleanText(text2)
text_vec2 = tokenize_text(text2)

vec_test = model.infer_vector(text_vec)
vec_test2 = model.infer_vector(text_vec2)
print(min(vec_test))
