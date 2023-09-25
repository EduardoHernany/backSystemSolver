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
                matrix = np.array(matrix)

                b = matrix[:, -1]
                # Separe 'A' do restante da matriz
                A = matrix[:, :-1]

                A = A.astype('float64')
                b = b.astype('float64')
                x, A, b,steps= elimination_gaussianaa(A,b)

                # Concatene 'A' e 'b' horizontalmente
                augmented_matrix = np.column_stack((A, b))



                # Retorne a matriz como JSON
                return JsonResponse({'result': x.tolist(),'matrix': augmented_matrix.tolist(),'pasos':steps})
            else:
                return JsonResponse({'error': 'Dados de matriz ausentes ou em formato inválido.'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Falha ao analisar os dados JSON.'}, status=400)
    else:
        return JsonResponse({'error': 'Método não permitido'}, status=405)

def elimination_gaussianaa(A, b):
    n = len(b)
    x = np.zeros(n)
    steps = ""

    # Fase de eliminação
    for i in range(n):
        pivot = A[i, i]
        if pivot == 0:
            return None  # Verificar divisão por zero
        
        # Adicione o passo à string
        steps += f"Passo {i + 1}:\n"
        steps += f"Pivot = {pivot}\n"
        steps += f"Dividir a linha {i + 1} por {pivot}\n"

        for j in range(i + 1, n):
            factor = A[j, i] / pivot
            steps += f"Subtrair {factor:.2f} vezes a linha {i + 1} da linha {j + 1}\n"

            b[j] -= factor * b[i]
            A[j, i:] -= factor * A[i, i:]
    
    # Fase de substituição retroativa
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x, A, b,steps

def elimination_gaussiana(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            matrix_a_data = data.get('matrixA')
            vector_b_data = data.get('vectorB')

            # Converter os dados da matriz 'A' e do vetor 'b' em arrays NumPy
            matrix_a = np.array(matrix_a_data, dtype=float)
            vector_b = np.array(vector_b_data, dtype=float)

            n = len(vector_b)

            # Etapa de Eliminação
            for pivot_row in range(n):
                # Encontre o pivô (o elemento principal da linha)
                pivot = matrix_a[pivot_row][pivot_row]

                # Divida a linha do pivô pelo seu valor para fazer o pivô igual a 1
                matrix_a[pivot_row] /= pivot
                vector_b[pivot_row] /= pivot

                # Subtraia a linha do pivô das linhas abaixo dela para zerar os elementos abaixo do pivô
                for row in range(pivot_row + 1, n):
                    factor = matrix_a[row][pivot_row]
                    matrix_a[row] -= factor * matrix_a[pivot_row]
                    vector_b[row] -= factor * vector_b[pivot_row]

            # Etapa de Substituição Retroativa
            x = np.zeros(n)
            for i in range(n - 1, -1, -1):
                x[i] = vector_b[i]
                for j in range(i + 1, n):
                    x[i] -= matrix_a[i][j] * x[j]

            # Retorne o resultado
            return JsonResponse({'result': x.tolist()})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Falha ao analisar os dados JSON.'}, status=400)
    else:
        return JsonResponse({'error': 'Método não permitido'}, status=405)