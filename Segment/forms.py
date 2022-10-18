import pandas as pd

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import StringField, SelectField, FieldList, FormField, Form, RadioField
from werkzeug.utils import secure_filename

class UploadForm(FlaskForm):
    file = FileField(validators=[FileRequired()])
    enc = StringField(default="UTF-8")

class EditVariableForm(Form):
    field_name = StringField('variable name')
    select = SelectField(
        'Select',
        choices=[
            ('num','Numeric'),
            ('cat', 'Categorical'),
            ('str', 'Text'),
            ('dt', 'Datetime')
        ]
    )

class EditTableForm(FlaskForm):
    variable_fields = FieldList(FormField(EditVariableForm), min_entries=1)
    
class ManipulateForm(FlaskForm):
    dummy = StringField('dummy')    

class ClusterForm(FlaskForm):
    cluster_alg = SelectField(
        'Cluster Method',
        choices=[
            ('KMeans','K-Means'),
            ('DBSCAN','DBSCAN'),
            ('AgglomerativeClustering','Agglomerative Clustering')
        ]
    )
    use_tsne = RadioField(
        'Use TSNE?',
        choices=[
            ('Yes','Yes'),
            ('No','No')
        ],
        default="Yes"
    )