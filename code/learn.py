# coding:utf-8
import threading
import time


class Sen(object):
        
    def sentenceInSetByPeopelGraphResult(sen):
        graph_result = [[['impuestos']],[['Cómo'],['reporto','enviar','informar','reportar','informo'],['proveedor']],[['hacer','Cómo'],['pedido']],[['bancaria']],[['Quiero'],['pagar']], [['no','ni','nunca'],['pedido']], [['Donde'],['cupones']],[['número'],['teléfono']],[['Recibí'],['pedido']],[['recibí','recibido'],['no','ni','nunca']],[['confiable'],['vendedor','proveedor']],[['protección'],['comprador','compra']],[['mi'],['pregunta']]]
        words = sen.lower().strip('¿').split()
        for sub_graph in graph_result:
            flag = 0
            for andSection in sub_graph:
                for orWord in andSection:
                    if orWord.lower() in words:
                        flag += 1
                        break
            if flag == len(sub_graph):
                return True
        return False

sentence = '¿Cuántos Impuestos pagar?'
print(Sen.sentenceInSetByPeopelGraphResult(sentence))