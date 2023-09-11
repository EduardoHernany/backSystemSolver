from django.shortcuts import render

from rest_framework import generics
from rest_framework.response import Response
from .models import Matrix
from .serializers import MatrixSerializer
import numpy as np


def triangularize_matrixx(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    for i in range(rows):
        # Encontre a primeira linha com um valor não zero na coluna atual
        pivot_row = i
        while pivot_row < rows and matrix[pivot_row][i] == 0:
            pivot_row += 1

        # Se não encontrarmos uma linha com valor não zero, pule para a próxima coluna
        if pivot_row == rows:
            continue

        # Troque as linhas para mover o pivô para a posição atual
        matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]

        # Normalize a linha atual para que o elemento de pivô seja 1
        pivot_element = matrix[i][i]
        for j in range(cols):
            matrix[i][j] /= pivot_element

        # Elimine os elementos abaixo do pivô
        for k in range(i + 1, rows):
            factor = matrix[k][i]
            for j in range(cols):
                matrix[k][j] -= factor * matrix[i][j]

    return matrix

# django_app/views.py

from django.http import JsonResponse
import json

def triangularize_matrix(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            matrix = data.get('data')

            if matrix:
                


                # Suponha que você queira apenas retornar a matriz original por enquanto
                result = triangularize_matrixx(matrix)

                # Retorne a matriz de resultado como JSON
                return JsonResponse({'result': result})
            else:
                return JsonResponse({'error': 'Dados de matriz ausentes ou em formato inválido.'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Falha ao analisar os dados JSON.'}, status=400)
    else:
        return JsonResponse({'error': 'Método não permitido'}, status=405)


def gaussian_elimination(matrix):
    # Converter a matriz para um formato numpy
    matrix = np.array(matrix, dtype=float)

    n, m = matrix.shape
    if n + 1 != m:
        raise ValueError("A matriz deve ser do formato nxn+1 para representar um sistema linear.")

    # Eliminação Gaussiana
    for i in range(n):
        # Pivoteamento parcial (trocar linhas se necessário)
        max_row = i
        for k in range(i + 1, n):
            if abs(matrix[k, i]) > abs(matrix[max_row, i]):
                max_row = k
        matrix[[i, max_row]] = matrix[[max_row, i]]

        # Escalonar a linha atual
        for j in range(i + 1, n):
            ratio = matrix[j, i] / matrix[i, i]
            matrix[j, i:] -= ratio * matrix[i, i:]

    # Resolução de retrosubstituição
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (matrix[i, -1] - np.dot(matrix[i, i+1:n], x[i+1:])) / matrix[i, i]

    return x.tolist()