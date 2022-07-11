from django.shortcuts import render
from pygments.unistring import Pe
from nltk.util import pr

# Create your views here.

def mainFunc(request):
    return render(request, 'main.html')

def page1Func(request):
    return render(request, 'page1.html')

def page2Func(request):
    return render(request, 'page2.html')

def cartFunc(request):
    # 신청한 제품을 세션에 담기.
    # name = request.POST.get('name')
    name = request.POST['name']
    price = request.POST['price']
    product={'name':name, 'price':price}
    
    productList =[]
    
    if 'shop' in request.session:
        productList = request.session['shop']
        productList.append(product)
        request.session['shop'] = productList
    else:
        productList.append(product)
        request.session['shop'] = productList #shop이라는 세션이 없다면 프러덕트리스트에 shop이라는 key로 만들어줌
    
    print(productList)
    
    context = {}
    context['products'] =  request.session['shop']
    return render(request, 'cart.html', context)

def buyFunc(request):
    if 'shop' in request.session:
        productList =  request.session['shop']
        total = 0
        for p in productList:
            total += int(p['price'])
        print('총계: ', total)
        request.session.clear() # 세션 전부 삭제 / 특정 섹션은 del request.session['shop']
        
    return render(request, 'buy.html',{'total':total})