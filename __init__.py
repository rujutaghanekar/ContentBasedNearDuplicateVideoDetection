import os
from flask import Flask, request, redirect, url_for, flash, render_template
from flask_session import Session

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'E:/BE PROJECT/Flask/static/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4'])
filename = ""

app = Flask(__name__)
sess = Session()
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET KEY'] = 'dsknfglkdfmg0987087dfmgldg'
sess.init_app(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


import globals
from fastyolo import yoloCall

from Qfunc import VideoToFrames
from Qfunc import createFeatureVectors
from Qfunc import generateKeyFrames
from Qfunc import DatabaseVideoIDExtraction
from Qfunc import findSimilarityMatrix
from Qfunc import AutomateDBParameters
from Qfunc import createThumbnail
from Qfunc import deleteFrames



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/results')
def displayResult():
    global filename
    reqJsonGraph = globals.df_similarity.to_json()
    scatterPlotData = globals.df_scatterPlot.to_json()
    print(scatterPlotData)
    reqJson = {"vidName": globals.videoName+".mp4", "queryClusters": globals.n_clusters_q,
               "videosCompared": globals.dbCompared, "duration": globals.duration, "totalFrames": globals.count}
    return render_template('index.html', reqJson=reqJson,reqJsonGraph=reqJsonGraph,queryClusters=globals.n_clusters_q,videosCompared=globals.dbCompared,scatterPlotData = scatterPlotData)


@app.route('/')
@app.route('/index')
def index():
    return render_template('form_upload.html')


@app.route('/afterUpload', methods=['GET', 'POST'])
def beginProcessing():
    done = deleteFrames()
    globals.initialize()
    print(done)
    global filename
    filename = ""
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            processing()
    return redirect(url_for('displayResult'))



def processing():
    global filename

    result1 = VideoToFrames(UPLOAD_FOLDER+"/"+filename[:-4], filename[:-4])
    result2 = createFeatureVectors()
    result3 = AutomateDBParameters()
    result4 = generateKeyFrames()
    img = yoloCall()
    print(globals.keys)
    print(globals.values2)
    result5 = DatabaseVideoIDExtraction()
    result7 = createThumbnail(img)
    print(globals.df_storeImageThumbnailQuery)
    result6 = findSimilarityMatrix()
    print(result1,result2,result3,result4,result5,result7,result6)


@app.route('/detailedAnalysis')
def detailedAnalysis():
        #vidLen = 4.23
        #dough1=[[12,78,34],[67,5,23]]
        #reqJson = {"dough1":dough1,"vidLen":vidLen}
    global filename
    print(globals.df_similarity)
    reqJson = globals.df_similarity.to_json();
    print(globals.color)
    return render_template('detailedAnalysis.html', reqJson=reqJson,queryClusters=globals.n_clusters_q,videosCompared=globals.dbCompared,color=globals.color)


if __name__ == "__main__":
    globals.initialize()
    app.run(debug=True)
