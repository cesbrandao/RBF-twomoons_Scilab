clear;
clc;

dados = fscanfMat('twomoons.dat'); //[l c] = [1001 3]

q = 15;//quantidade de neuronios
p = 2;//quantidade de atributos de entrada

//1001 amostras
Z = zeros(q,1001);//(ativação)
centroide = rand(p,q,'normal');//centroide dos neurônios ocultos

//normalização dos dados com zscores
for i = 1:2
    dados(:,i) = (dados(:,i) - mean(dados(:,i)))/stdev(dados(:,i));
end

/////PLOTAGEM INICIAL DOS DADOS/////
X_classe1 = dados(1:500,1);
Y_classe1 = dados(1:500,2);
X_classe2 = dados(501:1001,1);
Y_classe2 = dados(501:1001,2);

plot(X_classe1,Y_classe1,'gd');
plot(X_classe2,Y_classe2,'bx');
////////////////////////////////////

//atribuição dos valores de X e Y após a normalização
X = dados(:,1:2)';//entradas (transposto)
Y = dados(:,3)';//classes (transposto)

//ativação
for j = 1:1001
    for k = 1:q
        Z(k,j) = exp(-norm(X(:,j) - centroide(:,k)).^2);
    end
end

Z_Bias = [(-1)*ones(1,1001);Z];//adição do Bias

output = Y*Z_Bias'*(Z_Bias*Z_Bias')^(-1);//vetor dos pesos dos neurônios de saída

label = output*Z_Bias;//rede treinada

/////GRÁFICO/////

//pontos referentes a saída do gráfico, 500 pontos para cada etapa da verificação
pX = linspace(-2.3,2.3,500);
pY = linspace(-2.3,2.3,500);

//verificação da rede treinada em um conjunto de pontos X e Y
for l = 1:500
    for m = 1:500
        for n = 1:q
            pZ(n) = exp(-norm([pX(l) pY(m)]' - centroide(:,n)).^2);//ativação dos centróides
        end
        
        pLabel = output*[-1; pZ];//rótulos de saída de teste
        
if pLabel < 0.001 & pLabel > -0.001 then //precisão
            plot(pX(l),pY(m),'k.');
        end
    end
end
