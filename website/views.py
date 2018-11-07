from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import dialogflow, json, requests
from google.protobuf.json_format import MessageToJson

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="TaipeiBus-d9a21c23d606.json"
DIALOGFLOW_PROJECT_ID = 'taipeibus-3f5d4'

def index(request):
    title = 'main'
    return render(request, 'website/index.html', locals())

def camera(request):
    title = 'camera'
    return render(request, 'website/camera_test.html', locals())
    
@csrf_exempt
def uploadImage(request):
    if request.method == 'POST':
        print('uploadImage start...')
        imgString = request.POST['image']
        imgString = imgString.replace('data:image/png;base64,', '')

        import base64
        imgdata = base64.b64decode(imgString)
        filename = 'cameraCapture.png'  
        with open(filename, 'wb') as f:
            f.write(imgdata)
        print('uploadImage end...')
        return HttpResponse('uploadImage done')

def chatbot(request):
    title = 'chatbot'
    return render(request, 'website/chatbot.html', locals())

# 接Google Dialogflow 語意分析參數
@csrf_exempt
def webhook(request):
    if request.method == 'POST':
        # print(json.loads(request.body).keys())
        # 取得公車路線號碼
        bus_route = json.loads(request.body)['queryResult']['parameters']['bus_route_number']
        bus_route = str(int(bus_route))
        print(bus_route)
        # 設定公共運輸API需要的資料
        payload = {
            '$format': 'JSON'
        }
        # 僞裝request來自於瀏覽器，不需要申請API KEY
        header = {'user-agent':'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'}
        # 送出要求
        r = requests.get('https://ptx.transportdata.tw/MOTC/v2/Bus/RealTimeNearStop/City/Taipei/' + bus_route, params=payload, headers=header)
        print(r.url)
        # 要求成功
        if r.status_code == 200:
            # 讀出資料
            body = r.json()
            print(body)

            # 打包回給google dialogflow的fulfillmentMessage
            fulfillmentMessagesObj = []
            for bus in body:
                direction = ""
                if bus['Direction'] == 0:
                    direction = '去程'
                elif bus['Direction'] == 1:
                    direction = '返程'
                # 顯示該路線的公車車牌、方向（去程或返程）、站牌名稱
                fulfillment_text = {
                    "text":{
                        "text":[bus["PlateNumb"], "direction: " + direction, "stop: " + bus['StopName']['Zh_tw']]
                    }
                }
                fulfillmentMessagesObj.append(fulfillment_text)

            # 傳JSON回給google dialogflow
            response = JsonResponse({'fulfillmentMessages':fulfillmentMessagesObj})
            return response

def detect_intent_texts(project_id, session_id, text, language_code):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)

    if text:
        text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.types.QueryInput(text=text_input)
        response = session_client.detect_intent(session=session, query_input=query_input)
        print(response.query_result)
        return response.query_result.fulfillment_messages

@csrf_exempt
def send_message(request):
    if request.method == 'POST':
        message = request.POST['message']
        project_id = DIALOGFLOW_PROJECT_ID
        fulfillment_text = detect_intent_texts(project_id, "unique", message, 'en')
        print(type(fulfillment_text))
        fulfillment_obj = []
        for obj in fulfillment_text:
            fulfillment_obj.append(json.loads(MessageToJson(obj)))

        # fulfillment_text = json.loads(message_json)
        response_text = { "message":  fulfillment_obj }
        print(response_text)
        return JsonResponse(response_text)