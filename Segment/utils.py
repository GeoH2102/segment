import json
import re

import numpy as np
import pandas as pd

from plotly import graph_objects as go, utils as pltutils, express as px

def create_html_output(df, div_size, form=None, render_scrolls=False):
    html_output = df.sample(20).to_html(justify='left', col_space=50, index=False, classes="table", table_id="pandas_data_table")
    select_code = []
    if form:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                dtype = 'num'
            elif pd.api.types.is_categorical_dtype(df[col]):
                dtype = 'cat'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                dtype = 'dt'
            else:
                dtype = 'str'

            select_html = next((i.select for i in form.variable_fields if i.label.text == col))
            dropdown_cell = f"""
                <td>
                    {select_html}
                </td>
            """
            select_code.append(dropdown_cell)

    graph_code = []
    margin = 15
    margins = {i: margin for i in ['l','r','b','t']}
    margins['pad'] = 20
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # print('Number: Output box + whisker')
            newdf = df.copy()[col]
            fig = px.box(newdf, y=col, width=div_size, height=div_size)
            fig.update_layout(
                autosize=False,
                margin=margins,
                showlegend=False,
                yaxis={'visible': False, 'showticklabels': False},
                xaxis={'visible': False, 'showticklabels': False}
            )
            graphJSON = f"""
                <td>
                    <div id="chart-{col}" class="chart"></div>
                    <script type='text/javascript'>
                        var graphs = {json.dumps(fig, cls=pltutils.PlotlyJSONEncoder)};
                        Plotly.plot('chart-{col}', graphs, {'{}'});
                    </script>
                </td>
            """
            graph_code.append(graphJSON)
        elif pd.api.types.is_categorical_dtype(df[col]):
            # print('Categorical: Output bar chart')
            newdf = df.copy()
            cats = newdf[col].value_counts()
            newdf[col] = np.where(newdf[col].isin(cats[9:].index), 'Other', newdf[col])
            newdf = newdf[col].value_counts()
            fig = px.bar(newdf, width=div_size, height=div_size)
            fig.update_layout(
                autosize=False,
                margin=margins,
                showlegend=False,
                yaxis={'visible': False, 'showticklabels': False},
                xaxis={'visible': False, 'showticklabels': False}
            )
            graphJSON = f"""
                <td>
                    <div id="chart-{col}" class="chart"></div>
                    <script type='text/javascript'>
                        var graphs = {json.dumps(fig, cls=pltutils.PlotlyJSONEncoder)};
                        Plotly.plot('chart-{col}', graphs, {'{}'});
                    </script>
                </td>
            """
            graph_code.append(graphJSON)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # print('Datetime: Output time series')
            newdf = df.copy()[col].dt.strftime('%Y-%V').value_counts().sort_index()
            fig = px.line(newdf, height=div_size, width=div_size)
            fig.update_layout(
                autosize=False,
                margin=margins,
                showlegend=False,
                yaxis={'visible': False, 'showticklabels': False},
                xaxis={'visible': False, 'showticklabels': False}
            )
            graphJSON = f"""
                <td>
                    <div id="chart-{col}" class="chart"></div>
                    <script type='text/javascript'>
                        var graphs = {json.dumps(fig, cls=pltutils.PlotlyJSONEncoder)};
                        Plotly.plot('chart-{col}', graphs, {'{}'});
                    </script>
                </td>
            """
            graph_code.append(graphJSON)
        else:
            # print('Text: Do not output')
            graph_code.append(f'<td><div style="height:{div_size}px; width:{div_size}px"></div></td>')
    if render_scrolls:
        select_code = '<tr>' + ''.join(select_code) + '</tr>'
    else:
        select_code = ''
    graph_code = '<tr>' + ''.join(graph_code) + '</tr>'
    html_output = re.sub('<tbody>\n', f'<tbody>\n{select_code}\n{graph_code}', html_output)

    return html_output

def generate_graph(data, df):
    print(df.columns)
    if len(data) == 0:
        return '200'
    var1 = data['var1']
    var2 = data['var2']
    graph_type = data['graph_type']
    cluster = eval(data['cluster'].capitalize())

    if cluster:
        color='cluster'
    else:
        color=None

    if graph_type == 'scatter':
        fig = px.scatter(df, x=var1, y=var2, color=color)
        graphJSON = json.dumps(fig, cls=pltutils.PlotlyJSONEncoder)
    elif graph_type == 'boxplot':
        if pd.api.types.is_numeric_dtype(df[var1]):
            numvar = var1
            catvar = var2
        else:
            numvar = var2
            catvar = var1
        fig = px.box(df, x=catvar, y=numvar, color=color)
        graphJSON = json.dumps(fig, cls=pltutils.PlotlyJSONEncoder)
    elif graph_type == 'histogram':
        if pd.api.types.is_numeric_dtype(df[var1]):
            numvar = var1
            catvar = var2
        else:
            numvar = var2
            catvar = var1
        fig = px.histogram(df, x=catvar, y=numvar, histfunc='avg', color=color)
        graphJSON = json.dumps(fig, cls=pltutils.PlotlyJSONEncoder)
    elif graph_type == 'timeseries-num':
        if pd.api.types.is_datetime64_any_dtype(df[var1]):
            dtvar = var1
            numvar = var2
        else:
            dtvar = var2
            numvar = var1
        n_years = df[dtvar].dt.strftime('%Y').nunique()
        n_quarters = df[dtvar].dt.to_period('Q').nunique()
        n_months = df[dtvar].dt.strftime('%Y-M%m').nunique()
        n_weeks = df[dtvar].dt.strftime('%Y-W%V').nunique()
        if n_years >= 8:
            aggdf = df.groupby(df[dtvar].dt.strftime('%Y'))[numvar].mean()
        elif n_quarters >= 8:
            aggdf = df.groupby(df[dtvar].dt.to_period('Q'))[numvar].mean()
            aggdf.index = aggdf.index.astype(str)
        elif n_months >= 8:
            aggdf = df.groupby(df[dtvar].dt.strftime('%Y-M%m'))[numvar].mean()
        elif n_weeks >= 8:
            aggdf = df.groupby(df[dtvar].dt.strftime('%Y-W%V'))[numvar].mean()
        else:
            aggdf = df.groupby(df[dtvar].dt.strftime('%Y-D%j'))[numvar].mean()

        fig = px.line(aggdf.reset_index(), x=dtvar, y=numvar, color=color)
        graphJSON = json.dumps(fig, cls=pltutils.PlotlyJSONEncoder)
    elif graph_type == 'timeseries-cat':
        if pd.api.types.is_datetime64_any_dtype(df[var1]):
            dtvar = var1
            catvar = var2
        else:
            dtvar = var2
            catvar = var1
        n_years = df[dtvar].dt.strftime('%Y').nunique()
        n_quarters = df[dtvar].dt.to_period('Q').nunique()
        n_months = df[dtvar].dt.strftime('%Y-M%m').nunique()
        n_weeks = df[dtvar].dt.strftime('%Y-W%V').nunique()
        if n_years >= 8:
            aggdf = df[[dtvar, catvar]]
            aggdf[dtvar] = aggdf[dtvar].dt.strftime('%Y')
        elif n_quarters >= 8:
            aggdf = df[[dtvar, catvar]]
            aggdf[dtvar] = aggdf[dtvar].dt.to_period('Q').astype(str)
        elif n_months >= 8:
            aggdf = df[[dtvar, catvar]]
            aggdf[dtvar] = aggdf[dtvar].dt.strftime('%Y-M%m')
        elif n_weeks >= 8:
            aggdf = df[[dtvar, catvar]]
            aggdf[dtvar] = aggdf[dtvar].dt.strftime('%Y-W%V')
        else:
            aggdf = df[[dtvar, catvar]]
            aggdf[dtvar] = aggdf[dtvar].dt.strftime('%Y-D%j')

        if cluster:
            df[catvar] = df[catvar] + '-' + df['cluster'].astype(str)
        fig = px.histogram(df, x=dtvar, color=catvar)
        graphJSON = json.dumps(fig, cls=pltutils.PlotlyJSONEncoder)
        
    elif graph_type == 'heatmap':
        aggdf = pd.pivot_table(
            df[[var1, var2]].assign(n=1),
            index=var1,
            columns=var2,
            values='n',
            aggfunc='count'
        )
        fig = px.imshow(aggdf, text_auto=True, aspect='auto', color=color)
        graphJSON = json.dumps(fig, cls=pltutils.PlotlyJSONEncoder)

    return graphJSON