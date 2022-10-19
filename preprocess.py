import random
import re
import requests

import numpy as np
import pandas as pd

from IPython.display import display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

df = pd.read_csv('/home/george/Development/Personal/Python/Segment/data/data.csv', encoding='ISO-8859-1')
display(df)

# Colour
colours = (
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bred\b'), 'red',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bblue\b|\bteal\b|\bturquoise\b'), 'blue',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bwhite\b|\bivory\b'), 'white',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\byellow\b|\bcream\b'), 'yellow',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bgreen\b|\bjade\b|\bmint\b'), 'green',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bblack\b'), 'black',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bsilver\b|\bzinc\b|\bgold\b|\bmetal\b'), 'metallic',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bpurple\b'), 'purple',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\borange\b'), 'orange',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bgrey\b'), 'grey',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bwicker\b|\bwood\b|\bwooden\b'), 'wood',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bpink\b'), 'pink', 'n/a'))))))))))))
)

df['Colour'] = colours

# countvectorizer = CountVectorizer(
#     ngram_range=(1,1),
#     stop_words='english',
#     min_df=10
# )
# countmatrix = countvectorizer.fit_transform(df['Description'].dropna())
# count_df = pd.DataFrame(countmatrix.toarray())
# count_df.columns = countvectorizer.get_feature_names_out()
# term_sums = count_df.sum()
# term_sums.sort_values().tail(20)

# Product category
categories = (
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bbag\b|\bwashbag\b'), 'bag',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'christmas\b|\bnoel wooden\b|\beaster\b'), 'seasonal',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bsign\b|\bframe\b|\bdoorsign\b|\bwelcome\b|\bplaque\b|\bphotoframe\b|\bthermometer\b|\bflag of st george\b|\bcanvas screen\b|\bmirror\b|\bmiror\b'), 'signs & frames',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\blight\b|\blights\b|\blantern|candle|\bnightlight|\blightbulb\b|\bbulb\b'), 'lights',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bwater bottle\b|\bhottie\b'), 'hot water bottle',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bdolly\b'), 'doll',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'knob\b|\bdoormat\b|\bcoat rack\b|\bdrawer\b|\btissues\b|\bmagnetic\b|\bart\b|\bcabinet\b|\bshelf\b|\bhook\b|\bfob\b|\bumbrella\b|\bcushion\b|\bhome\b|\bset\b|\bphoto\b|\bdove\b|\bdoorstop\b|\btidy\b|\blipstick\b|\bmat\b|\bdrawer|\bbib\b|\brack\b|\bdoily\b|\bdoilies\b|\bsponge\b|\bmirror\b|\bhammock\b|\bbell\b|\bluggage\b|\bpostcard\b|\blamp\b|\bplace setting|\bcarriage\b|\bcardholder|\bhamper\b|\btablecloth\b|\btowel\b|\bfire bucket\b|\bplacemat|\bironing board\b|\bflannel\b|\bincense\b|\bjewellery\b|\bcurtain|\blampshade\b|\bcoaster\b|\bradio\b|\benglish rose\b|\bcushion cover\b|\bpouffe\b|\bfruitbowl\b|\bseat\b|\bsquarecushion\b|\bmobile\b|\bthrow\b|\bbullet bin\b|\btrinket\b|\bquilt\b|\bround table\b|\bwhite base\b|\bfolding chair\b|\bsteel table\b|\bstool\b|\bdesk and chair\b|\btissue ream\b|\bashtray\b|\bchandelier|\btable run flower\b|\bst george chair\b|\bnewspaper stand\b|\bkashmiri chair\b|\bpartition panel\b|\bdog collar\b|\bfragrance oils\b'), 'home',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\blunch box\b'), 'lunch box',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bclock\b'), 'clock',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bteacup\b|\bbowls\b|\bbowl\b|\bceramic\b|\bscales\b|\bcookie cutter|\bcake\b|\bcakestand\b|\bjelly\b|\bbaking\b|\bapron\b|\bplate|\bjug\b|\bmeasuring\b|\bbread bin\b|\boven\b|\btea\b|\begg\b|\bteapot|\bcutlery\b|\bbottle|\bcup\b|\bwashing\b|\bcases\b|\brecipe\b|\bmug\b|\bjar\b|\bdispenser\b|\bpantry\b|\bdish\b|\bpan\b|\btongs\b|\bstraws\b|\bmugs\b|\bcolander\b|\bfood cover\b|\bcloche\b|\btube match|\btray\b|\bmoulds\b|\btoast\b|\bflask\b|\bbeaker|\bbasin\b|\btoastrack\b|\bice cream\b|\bcocktail\b|\bkitchen\b|\bspoon\b|\bcoffee\b|\btumbler\b|\bcamphor wood\b|\bchopping board\b|\bbiscuit tin\b|\bglass\b|\bbiscuit bin\b|\bice lolly\b|\bliners\b|\borange squeezer\b|\bplatter\b|\bgoblet\b|\bjampot\b|\bcannister\b|\bsweetheart trays\b'), 'kitchen',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bvintage\b'), 'vintage',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bcard\b'), 'card',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bbunting\b|\bnapkins\b|\bribbon|\bpaper plate|\bpaper cup|\bballoon|\bparty\b|\bconfetti\b'), 'party',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bornament\b|\bfan\b|\bgarland\b|\bwood letters\b|\bpaper ball\b|\bpainted letter|\btoadstool|\bwreath\b|\bleis\b|\bblock letters\b|\bwooden daisy\b|\bpainted metal\b|\bpink rabbit\b|\bbunny\b|\bdisco ball\b|\bheart\b|\bwindmill\b|\bmagnet|\bmetal pears\b|\bfolkart\b|\bdecoration|\bporcelain\b|\blaurel star\b|\bknitted hen\b|\bglass chalice\b|\bfeather tree\b|\bscottie dog\b|\bfabric pony\b|\bclam shell\b|\bvase\b|\bartificial flower\b|\bartifcial\b|\bhand open shape\b|\bgingham cat\b|\bgeisha\b|\bmetal cat\b|\bmetal string\b|\bwindsock\b|\bfairy pole\b|\bbaby mouse\b|\bbuddha\b|\bacrylic jewel\b|\bjewel icicle\b|\brabbit\b|\bfloral \w+ monster\b|\btube chime\b|\bbutterfly\/crystal\b|\bstring of 8 butterflies\b|\bflowers pony\b|\bartificial flower\b|\bartiifcial\b|\btreasure chests\b|\bsilicon cube\b|\bsinging canary\b'), 'ornaments',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bchalkboard\b|\bblack board\b|\bmemo board\b|\bnotebook|\bstationery\b|\bgift\b|\bruler\b|\bchalk\b|\bwriting\b|\bwrap\b|\beraser|\bbook|\bcalendar\b|\benvelope|\bpen|\btape\b|\baddress\b|\bfiller pad\b|passport\b|\bwastepaper bin\b|\borganiser\b|\bjournal\b|\bsketchbook\b|\bpaperweight\b|\brubber\b|\bblackboard\b|\bfirst aid kit\b|\bp\'weight\b|\bc\/cover\b|\bhole punch\b'), 'office',
    # np.where(df['Description'].fillna('').str.lower().str.contains(r'\bpostage\b|\bcash\+carry\b|\bmanual\b|\bdiscount\b|\bsamples\b|\bcheck\b|^\?|\bdamages\b|\bdamaged\b|\bbank charges\b|\bamazon fee\b|\bfound\b|\bcommission\b|\badjustment|\bpacking charge\b|\bdestroyed\b|\bthrown away\b|\bamazon\b|\bdotcom\b|\bebay\b|\bput aside\b|\bpads to match\b|\bsmashed\b|\bhigh resolution image\b|\btest\b|\bmailout\b|\bmissing\b|\bwet pallet\b|\badjust bad debt\b|\bincorrectly credited\b|\bwet\/rusty\b|\bmixed up\b|\bwet rusty\b|\bwrongly coded\b|\bincorrect stock\b|\btaig adjust\b|\bsold as 1\b|\bcrushed\b|\bwrong\b|\boops\b|\bcounted\b|\bzero invc\b|\breturned\b|\balan hodge\b|\bstock creditted\b|\bstock\b|\bcode mix\b|\bonline retail\b|\bwebsite fixed\b|\bmix up\b|\bcame coded\b|\bfba\b|\bdagamed\b|\bhistoric computer\b|\bwet\/mouldy\b|\bcan\'t find\b|\bmouldy\b'), 'services',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bmaking\b|\bknitting\b|\bpaper chain\b|craft|\bsewing\b|\bclay\b|\btapes\b|\bscissor\b|\bpatches\b|\bsew on\b|\bpaint your own\b|\belectronic meter\b'), 'crafts',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bgliders\b|\bblock word\b|\bspinning tops\b|\btoy\b|\bsoldier\b|\bplayhouse\b|\bskipping rope\b|\bcrayon|\bcolouring\b|\bgame\b|\bbingo\b|\bkids\b|\bpencil|\bpicture|\balphabet\b|\bpiggy bank\b|\blip gloss\b|\bplaying\b|\bspaceboy\b|\bnaughts\b|\bponcho\b|\bring\b|\bglitter\b|\bsandcastle\b|\bludo\b|\bfun\b|\brocking horse\b|\bchocolate\b|\bcircus\b|\bdinosaur\b|\btattoos\b|\bsticky\b|\bteddy\b|\bmagic drawing\b|\bjigsaw\b|\bstickers\b|\blolly maker|\bmagic sheep\b|\bmagic tree\b|\bcreepy crawlies\b|\bsnake eggs\b|\bfelt farm\b|\bchildren\b|\bseaside\b|\bsticker sheet\b|\bglobe\b|\bspace owl\b|\bhelicopter\b|\bsock puppet\b|\bninja\b|\bspace frog\b|\bspace cadet\b|\bcluster slide\b|\bcinderella\b|\bglow in dark\b|\bdoll\b|\binflatable\b|\bfarmyard animals\b|\bstress ball\b|\bphone char\b|\bceature screen\b|\bipod\b'), 'kids',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bponcho\b|\bhat\b|\bhand warmer\b|\bwallet\b|\bpurse\b|\bbackpack\b|\bbracelet\b|\bslipper|\bhandbag\b|\bearring|\bhairclip|\bsombrero\b|\bcomb\b|\bkeyring\b|\bphone charm\b|\bgrip\b|\bear muff\b|\bnecklace\b|\bhair band\b|\bbrooch\b|\brosette\b|\bskirt\b|\btiara|\bhair tie\b|\bbangle\b|\bhairslide\b|\bshower cap\b|\bpointy shoe\b|\bhair clip|\blariat\b|\bsunglasses\b|\bnecklac|\bbraclet|\b42\"neckl|\bkey-chains\b|\brucksack\b|\bdiamante chain\b|\bglasses case\b'), 'clothing & accessories',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bgardeners\b|\bgarden\b|\bparasol\b|\btrellis\b|\bwatering\b|\bhanging\b|\bherb\b|\bwicker\b|\bspade\b|\bscrewdriver\b|\btorch\b|\brepair\b|\bspirit level\b|\bshed\b|\bbird house\b|\bbasket\b|\bbicycle clips\b|\bfly swat\b|\bgrow your own\b|\bplanter pots\b|\bplant cage\b|\bgnome\b|\bhen house\b|\bbird feeder\b|\bbird table\b|\bplanters\b|\bwindchime\b|\bfeeding station\b|\bdovecote\b'), 'garden & tools',
    np.where(df['Description'].fillna('').str.lower().str.contains(r'\bbox\b|\btin\b|\bboxes\b|\btins\b|\bholder\b|\bhangers\b|\bhanger\b|\bpegs\b|\bpot\b|\bchest\b'), 'storage', 'services')
))))))))))))))))))))

df['Category'] = categories
print(df.loc[df['Category'] == 'n/a', 'Description'].value_counts().sum())

# countvectorizer = CountVectorizer(
#     ngram_range=(2,2),
#     stop_words='english'
# )
# df_na = df.loc[df['Category']=='n/a','Description'].dropna()
# countmatrix = countvectorizer.fit_transform(df_na)
# count_df = pd.DataFrame(countmatrix.toarray())
# count_df.columns = countvectorizer.get_feature_names_out()
# term_sums = count_df.sum()
# display(term_sums.sort_values(ascending=False).head(10))

# display(df.loc[df['Category'] == 'n/a', 'Description'].value_counts().head(10))


# Create customer df
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['LineSum'] = df['Quantity'] * df['UnitPrice']

# Test word clusters for categories
# embeddings_dict = {}
# with open('models/glove.6B.50d.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:], "float32")
#         embeddings_dict[word] = vector

# def embedding(row):
#     if not isinstance(row, str):
#         row = ''
#     txt = row.lower()
#     txt = re.sub('\W', ' ', txt)
#     txt_array = txt.split()
#     embedding_list = []
#     for word in txt_array:
#         embedding_list.append(embeddings_dict.get(word, np.zeros(50)))
#     embedding_array = np.array([np.array(xi) for xi in embedding_list])
#     average_array = np.mean(embedding_array, axis=0)
#     return average_array

# working_col = df['Description']
# final_embeddings = working_col.apply(embedding)
# final_embeddings = final_embeddings.apply(lambda d: d if isinstance(d, np.ndarray) else np.zeros(50, dtype=np.float32))

# km = KMeans(n_clusters=8)
# kmout = km.fit_predict(final_embeddings.apply(pd.Series).to_numpy())
# df['Description_KM'] = kmout


# Drop customers who have returned 100%
drop_custs = df.groupby('CustomerID')['LineSum'].agg('sum')
drop_custs = drop_custs.loc[drop_custs <= 0]
df = df.loc[~df['CustomerID'].isin(drop_custs.index)].copy(deep=True)

idf = df.groupby(['CustomerID','InvoiceNo']).agg(
    {
        'LineSum': 'sum',
        'Quantity': 'sum'
    }
).reset_index()
idf = idf.groupby('CustomerID').agg(
    {
        'LineSum': 'mean',
        'Quantity': 'mean'
    }
).reset_index()
idf.columns = ['CustomerID','AvgOrderValue','AvgItemsPerOrder']


df['rowid'] = df.index
def pivot_df(df, vals, index, cols, prefix):
    pt = df.pivot_table(vals, index, cols).fillna(0)
    pt.index.name = None
    pt.columns.name = None
    pt.columns = [prefix + i for i in pt.columns]
    return pt

colours_quant_df = pivot_df(df, 'Quantity', 'rowid', 'Colour', 'quant-col-')
colours_spend_df = pivot_df(df, 'LineSum', 'rowid', 'Colour', 'spend-col-')
product_quant_df = pivot_df(df, 'Quantity', 'rowid', 'Category', 'quant-cat-')
product_spend_df = pivot_df(df, 'Quantity', 'rowid', 'Category', 'spend-cat-')

pdf = (
    df
        .merge(colours_quant_df, left_index=True, right_index=True)
        .merge(colours_spend_df, left_index=True, right_index=True)
        .merge(product_quant_df, left_index=True, right_index=True)
        .merge(product_spend_df, left_index=True, right_index=True)
)

pdf_aggs = {
    'InvoiceNo': 'nunique', # Number of orders
    'StockCode': 'nunique', # Unique items ordered
    'Quantity': 'sum', # Number of items ordered
    'LineSum': [
        'sum',
        lambda x: x.loc[x >= 0].sum(),
        lambda x: x.loc[x < 0].sum()
    ], # Total spend, Total ordered, Total returned
    'InvoiceDate': 'min' # First order
}
for i in pdf.columns:
    if ('-cat-' in i) or ('-spend-' in i):
        pdf_aggs[i] = 'sum'

cdf = pdf.groupby('CustomerID').agg(
    pdf_aggs
)
cdf.index.name = None
cdf.columns = [
    'Num_Orders', 'Unique_Items', 'Total_Items', 'Spend', 'Spend_Exc_Returns', 'Total_Returns',
    'FirstOrder', 'quant-cat-bag', 'quant-cat-card', 'quant-cat-clock', 
    'quant-cat-clothing & accessories', 'quant-cat-crafts', 'quant-cat-doll', 
    'quant-cat-garden & tools', 'quant-cat-home', 'quant-cat-hot water bottle', 'quant-cat-kids',
    'quant-cat-kitchen', 'quant-cat-lights', 'quant-cat-lunch box', 'quant-cat-office',
    'quant-cat-ornaments', 'quant-cat-party', 'quant-cat-seasonal', 'quant-cat-services',
    'quant-cat-signs & frames', 'quant-cat-storage', 'quant-cat-vintage', 'spend-cat-bag',
    'spend-cat-card', 'spend-cat-clock', 'spend-cat-clothing & accessories', 'spend-cat-crafts',
    'spend-cat-doll', 'spend-cat-garden & tools', 'spend-cat-home', 'spend-cat-hot water bottle',
    'spend-cat-kids', 'spend-cat-kitchen', 'spend-cat-lights', 'spend-cat-lunch box', 
    'spend-cat-office', 'spend-cat-ornaments', 'spend-cat-party','spend-cat-seasonal',
    'spend-cat-services', 'spend-cat-signs & frames', 'spend-cat-storage','spend-cat-vintage'
]
finaldf = cdf.merge(idf, left_index=True, right_on='CustomerID')

# Average Freq
freqdf = (
    df
        .copy()
        .sort_values(by=['CustomerID','InvoiceDate'], ascending=[True,True])
        .groupby(['CustomerID','InvoiceNo'])
        .agg({'InvoiceDate': max})
        .reset_index()
)
freqdf['next_InvoiceDate'] = (
    freqdf
        .groupby('CustomerID')['InvoiceDate']
        .transform(
            lambda x: x.shift(-1)
        )
)
freqdf['Avg_Freq'] = (freqdf['next_InvoiceDate'].dt.normalize() - freqdf['InvoiceDate'].dt.normalize()).dt.days
avgfreq = freqdf.groupby('CustomerID')['Avg_Freq'].agg('mean')

finaldf = finaldf.merge(avgfreq.to_frame(), left_on='CustomerID', right_index=True, how='left')
finaldf['Avg_Freq'] = finaldf['Avg_Freq'].fillna(finaldf['Avg_Freq'].max())

# Random category
finaldf['Cat'] = random.choices(['A','B','C','D'],k=len(finaldf))

# Random text
word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
response = requests.get(word_site)
WORDS = response.content.splitlines()
WORDS = [i.decode() for i in WORDS]

finaldf['Description'] = (finaldf.apply(
    lambda _: ' '.join(random.choices(WORDS, k=15)),
    axis=1
))


finaldf.to_csv('/home/george/Development/Personal/Python/Segment/data/processed.csv', index=False)