import json
import os
import re

import numpy as np
import pandas as pd

from collections import namedtuple
from flask import Flask, render_template, request, redirect
from flask_wtf.csrf import CSRFProtect
from plotly import graph_objects as go, utils as pltutils, express as px
from wtforms import StringField

from Segment.Segment import SegmentData
from Segment.constants import *
from Segment.forms import *
from Segment.utils import create_html_output, generate_graph

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(32)
csrf = CSRFProtect(app)
    
Segment = SegmentData(None)

@app.route('/', methods=['GET','POST'])
def home():

    form = UploadForm()

    if form.validate_on_submit():
        f = form.file.data
        enc = form.enc.data
        df = pd.read_csv(f, encoding=enc)
        Segment.df = df
        Segment.generate_manip_options()
        
        return redirect(
            '/change-dtypes'
        )

    return render_template('index.html', dfshape=Segment.df.shape, form=form)

@app.route("/change-dtypes", methods=["GET","POST"])
def change_dtypes():

    var_list = namedtuple('Field', ['field_name','select'])
    data = {
        'variable_fields': [
            var_list(i, None) for i in Segment.df.columns
        ]
    }

    form = EditTableForm(data=data)
    for n in range(len(form.variable_fields)):
        colname = form.variable_fields[n].field_name.data
        if pd.api.types.is_numeric_dtype(Segment.df[colname]):
            dtype = 'num'
        elif pd.api.types.is_categorical_dtype(Segment.df[colname]):
            dtype = 'cat'
        elif pd.api.types.is_datetime64_any_dtype(Segment.df[colname]):
            dtype = 'dt'
        else:
            dtype = 'str'
        form.variable_fields[n].select.data = dtype
        form.variable_fields[n].label.text = colname
        Segment._edit_data_mapping[form.variable_fields[n].name+'-select'] = colname
    if form.validate_on_submit():
        if request.form['submit'] == 'btn_change_dtypes':
            f = request.form
            for item in f:
                if item.startswith('variable_fields'):
                    colname = Segment._edit_data_mapping[item]
                    new_dtype = f.get(item)
                    if pd.api.types.is_numeric_dtype(Segment.df[colname]):
                        dtype = 'num'
                    elif pd.api.types.is_categorical_dtype(Segment.df[colname]):
                        dtype = 'cat'
                    elif pd.api.types.is_datetime64_any_dtype(Segment.df[colname]):
                        dtype = 'dt'
                    else:
                        dtype = 'str'
                    print(colname, dtype, new_dtype)
                    if dtype != new_dtype:
                        if new_dtype == 'num':
                            Segment.edit_to_number(colname)
                        elif new_dtype == 'cat':
                            Segment.edit_to_cat(colname)
                        elif new_dtype == 'dt':
                            Segment.edit_to_datetime(colname)
                        elif new_dtype == 'str':
                            Segment.edit_to_str(colname)
            Segment.generate_manip_options()
            return redirect('/change-dtypes')
        elif request.form['submit'] == 'btn_edit_to_manipulate':
            return redirect('/manipulate')
            
    return render_template(
        'edit_data.html', 
        html_table=create_html_output(Segment.df, DIV_SIZE, form=form, render_scrolls=True),
        dfshape=Segment.df.shape,
        form=form
    )

@app.route("/manipulate", methods=["GET","POST"])
def manipulate():
    
    Segment.df = pd.read_csv('/home/george/Development/Personal/Python/Segment/data/processed.csv')
    Segment.edit_to_cat('Cat')
    Segment.edit_to_datetime('FirstOrder')
    Segment.edit_to_str('CustomerID')
    
    Segment.manip_cat_bins('Cat',{'A':'A','B':'B','C':'C','D':'E'})
    Segment.manip_dt_datediff('FirstOrder','2010-12-01')
    Segment.manip_num_normalize('Num_Orders')
    Segment.manip_num_trim('Num_Orders',15,0)
    Segment.manip_str_embeddings('Description')
    
    form = ManipulateForm()
    
    fields = Segment.get_dtypes()
    if 'btn_manip_normalize' in request.form.keys():
        col = request.form['select-variable']
        fields = Segment.get_dtypes()
        Segment.manip_num_normalize(col)
        return render_template(
            'manipulate.html',
            variables=fields,
            manipulations=Segment._manipulations,
            html_table=create_html_output(Segment.df, DIV_SIZE),
            dfshape=Segment.df.shape,
            form=form
        )

    if 'btn_manip_trim' in request.form.keys():
        col = request.form['select-variable']
        fields = Segment.get_dtypes()
        lower = int(request.form['manip_trim_lower'])
        upper = int(request.form['manip_trim_upper'])
        Segment.manip_num_trim(col, upper, lower)
        return render_template(
            'manipulate.html',
            variables=fields,
            manipulations=Segment._manipulations,
            html_table=create_html_output(Segment.df, DIV_SIZE),
            dfshape=Segment.df.shape,
            form=form
        )

    if 'btn_manip_bin' in request.form.keys():
        d = request.form.to_dict()
        d = {
            re.sub('manip_bin_','',k): v 
            for k,v in d.items() if k.startswith('manip_bin')
        }
        Segment.manip_cat_bins(request.form.get('select-variable'), d)
        fields = Segment.get_dtypes()
        Segment.generate_manip_options() # Regenerate the manip options as cats will have changed
        return render_template(
            'manipulate.html',
            variables=fields,
            manipulations=Segment._manipulations,
            html_table=create_html_output(Segment.df, DIV_SIZE),
            dfshape=Segment.df.shape,
            form=form
        )
    
    if 'btn_manip_datediff' in request.form.keys():
        col = request.form['select-variable']
        datediff = request.form['manip_datediff']
        Segment.manip_dt_datediff(col, datediff)
        fields = Segment.get_dtypes()
        return render_template(
            'manipulate.html',
            variables=fields,
            manipulations=Segment._manipulations,
            html_table=create_html_output(Segment.df, DIV_SIZE),
            dfshape=Segment.df.shape,
            form=form
        )
    
    if 'btn_manip_emb' in request.form.keys():
        col = request.form['select-variable']
        fields = Segment.get_dtypes()
        Segment.manip_str_embeddings(col)
        return render_template(
            'manipulate.html',
            variables=fields,
            manipulations=Segment._manipulations,
            html_table=create_html_output(Segment.df, DIV_SIZE),
            dfshape=Segment.df.shape,
            form=form
        )

    if 'btn_manipulate_to_visualise' in request.form.keys():
        return redirect('/visualise')

    return render_template(
        'manipulate.html',
        variables=fields,
        manipulations=Segment._manipulations,
        html_table=create_html_output(Segment.df, DIV_SIZE),
        dfshape=Segment.df.shape,
        form=form
    )

@app.route("/visualise", methods=["GET","POST"])
def visualise():

    form = ManipulateForm()

    if 'btn_viz_to_cluster' in request.form.keys():
        return redirect('/cluster')

    graphJSON = generate_graph(request.args, Segment.df)
    return render_template(
            'visualise.html',
            graphJSON=graphJSON,
            dtypes=Segment.get_dtypes(),
            dfshape=Segment.df.shape,
            form=form
        )


@app.route("/callback", methods=["GET","POST"])
def cb():
    return generate_graph(request.args, Segment.df)

@app.route("/cluster", methods=["GET","POST"])
def cluster():

    Segment.df = pd.read_csv('/home/george/Development/Personal/Python/Segment/data/processed.csv')
    Segment.edit_to_cat('Cat')
    Segment.edit_to_datetime('FirstOrder')
    Segment.edit_to_str('CustomerID')
    
    Segment.manip_cat_bins('Cat',{'A':'A','B':'B','C':'C','D':'E'})
    Segment.manip_dt_datediff('FirstOrder','2010-12-01')
    Segment.manip_num_normalize('Num_Orders')
    Segment.manip_num_trim('Num_Orders',15,0)
    Segment.manip_str_embeddings('Description')

    form = ClusterForm()

    if form.validate_on_submit():
        if 'btn_cluster_to_clusterviz' in request.form.keys():
            return redirect('/clusterviz')

    return render_template(
        'cluster.html',
        dtypes=Segment.get_dtypes(),
        clustered=Segment._is_clustered,
        dfshape=Segment.df.shape,
        html_table=create_html_output(Segment.df, DIV_SIZE),
        form=form
    )

@app.route("/cluster-data", methods=["GET","POST","PUT"])
def cluster_data():
    Segment._cluster_fields = request.get_json()
    Segment.cluster(Segment._cluster_fields, n_components=3)
    return "ok!"

@app.route("/clusterviz")
def clusterviz():

    graphJSON = generate_graph(request.args, Segment.df)
    return render_template(
        'cluster_viz.html',
        graphJSON=graphJSON,
        dtypes=Segment.get_dtypes(),
        clustered=Segment._is_clustered,
        dfshape=Segment.df.shape
    )


@app.route("/test", methods=["GET","POST"])
def test():
    Segment.df = pd.DataFrame([
        {
            'abc': 'a b',
            'def': 'b',
            'ghi': 1
        },
        {
            'abc': 'd',
            'def': 'w',
            'ghi': 2
        },
        {
            'abc': 'a b',
            'def': 'w',
            'ghi': 2
        }
    ])
    Segment.df['abc'] = pd.Categorical(Segment.df['abc'])
    Segment.generate_manip_options()

    fields = Segment.get_dtypes()

    if 'btn_manip_normalize' in request.form.keys():
        col = request.form['select-variable']
        fields = Segment.get_dtypes()
        Segment.manip_num_normalize(col)
        return render_template(
            'manipulate.html',
            variables=fields,
            manipulations=Segment._manipulations,
            html_table=create_html_output(Segment.df, DIV_SIZE)
        )

    if 'btn_manip_trim' in request.form.keys():
        col = request.form['select-variable']
        fields = Segment.get_dtypes()
        lower = int(request.form['manip_trim_lower'])
        upper = int(request.form['manip_trim_upper'])
        Segment.manip_num_trim(col, upper, lower)
        return render_template(
            'manipulate.html',
            variables=fields,
            manipulations=Segment._manipulations,
            html_table=create_html_output(Segment.df, DIV_SIZE)
        )

    if 'btn_manip_bin' in request.form.keys():
        d = request.form.to_dict()
        d = {
            re.sub('manip_bin_','',k): v 
            for k,v in d.items() if k.startswith('manip_bin')
        }
        Segment.manip_cat_bins(request.form.get('select-variable'), d)
        fields = Segment.get_dtypes()
        return render_template(
            'manipulate.html',
            variables=fields,
            manipulations=Segment._manipulations,
            html_table=create_html_output(Segment.df, DIV_SIZE)
        )

    return render_template(
        'manipulate.html',
        variables=fields,
        manipulations=Segment._manipulations,
        html_table=create_html_output(Segment.df, DIV_SIZE)
    )


if __name__ == '__main__':
    app.run(debug=True)
