% =========================================================================
% SCRIPT DE PROCESAMIENTO, HISTÉRESIS Y FUSIÓN MULTI-CAPA (Exp 3)
% -------------------------------------------------------------------------
% Descripción: Este script orquesta el análisis simultáneo de sensores de 
% humedad ubicados en tres cotas espaciales (Inferior, Media, Superior).
% Extrae el canal Rojo (RGB_R) y gestiona la histéresis mediante la 
% separación algorítmica de las fases de absorción y desorción. 
% 
% Incorpora un sistema de 'Flags' booleanos para el control selectivo del 
% renderizado gráfico, evitando la saturación de memoria durante la 
% validación. Finalmente, fusiona las tres capas para generar un modelo 
% de "Sensor Ideal" robusto frente a gradientes de humedad verticales.
%
% Autor: Pedro Gabriel Fernández Cañete
% Institución: Universidad de Granada (UGR)
% =========================================================================

close all;
clc;
clear all;

%% 1. CONFIGURACIÓN INICIAL Y BANDERAS DE CONTROL (FLAGS)
% Rutas relativas apuntando al subdirectorio 'Datos/'
archivos_csv = ["Datos/datos_color_3_bot.csv", "Datos/datos_color_3_mid.csv", "Datos/datos_color_3_up.csv"];
nombres_cota = ["Inferior (Bot)", "Media (Mid)", "Superior (Up)"];

activate_RC = 1; % Flag (1/0): Calibración Radiométrica
ts = 2;          % Periodo de muestreo (2 segundos para Exp 3)
extraction_mode = 'RGB_R'; % Canal objetivo para medir la degradación por humedad

% --- SISTEMA DE RENDERIZADO GRÁFICO SELECTIVO ---
% Cambia a 'true' o 'false' según lo que desees visualizar en cada ejecución
mostrar_graficas.bot = false;    % Muestra las gráficas de la capa inferior
mostrar_graficas.mid = true;   % Muestra las gráficas de la capa media
mostrar_graficas.up  = false;   % Muestra las gráficas de la capa superior
mostrar_graficas.individuales = true; % Gráfica extra por cada una de las 4 ventanas

%% 2. INGESTA DE TELEMETRÍA (CÁMARA CLIMÁTICA)
memmert_filename = "Datos/TFG_hum.xls";
opts = detectImportOptions(memmert_filename, 'NumHeaderLines', 10);
opts = setvartype(opts, 'char'); 
memmert_data = readtable(memmert_filename, opts);

% 5 = Columna de Humedad Relativa (%)
columna_objetivo = 5; 
texto_variable = strrep(memmert_data{:, columna_objetivo}, ',', '.');
target_memmert = str2double(texto_variable);

duracion_ensayo_minutos = length(target_memmert) - 1;
time_memmert = (0:duracion_ensayo_minutos)';

%% 3. PREPARACIÓN TOPOLÓGICA Y EJE DE HUMEDAD COMÚN (HISTÉRESIS)
% Lectura de metadatos desde la primera capa para dimensionar matrices
tabla_temp = readtable(archivos_csv(1));
windowLabels = unique(tabla_temp.Window);
sensorLabels = windowLabels(~ismember(windowLabels, {'W_Ref', 'B_Ref'})); 

num_images = length(unique(tabla_temp.Image));
time = (0:1:num_images-1) .* ts;

% Sincronización del tiempo fotográfico con la telemetría
hum_interpolada = interp1(time_memmert, target_memmert, (time/60), 'linear', 'extrap');
idx_fin = find((time/60) <= duracion_ensayo_minutos, 1, 'last');

% Detección analítica de puntos de inflexión (Saturación y Secado)
[~, idx_pico] = max(hum_interpolada(1:idx_fin));
[~, idx_valle] = min(hum_interpolada(1:idx_pico));

% Generación del Eje Común de Alta Resolución (1000 muestras)
H_min = max(min(hum_interpolada(idx_valle:idx_pico)), min(hum_interpolada(idx_pico:idx_fin)));
H_max = min(max(hum_interpolada(idx_valle:idx_pico)), max(hum_interpolada(idx_pico:idx_fin)));
hum_comun = linspace(H_min, H_max, 1000); 

% Matriz 3D: [Cota(3) x Ventanas(4) x HumComun(1000)]
media_fases_all = zeros(3, length(sensorLabels), 1000);
colors = lines(length(sensorLabels));

%% 4. MOTOR DE PROCESAMIENTO MULTI-CAPA (Bucle Principal)
for s = 1:3
    dataTable = readtable(archivos_csv(s));
    Processed_Signal_Cota = zeros(length(sensorLabels), num_images);
    
    % Determinamos si la capa actual tiene permiso para generar gráficas
    renderizar = (s == 1 && mostrar_graficas.bot) || ...
                 (s == 2 && mostrar_graficas.mid) || ...
                 (s == 3 && mostrar_graficas.up);
    
    % --- 4.1 Aislamiento del Entorno Lumínico ---
    w_data = dataTable(strcmp(dataTable.Window, 'W_Ref'), :);
    b_data = dataTable(strcmp(dataTable.Window, 'B_Ref'), :);
    
    if renderizar
        figure('Name', sprintf('Referencias - Cota %s', nombres_cota(s)));
        subplot(2,1,1); plot(time, w_data.R, "r", time, w_data.G, "g", time, w_data.B, "b", "LineWidth", 2); grid on;
        title(sprintf("Reference: White (%s)", nombres_cota(s))); ylim([0 255]); xlabel("Time (s)"); ylabel("RGB Value");
        subplot(2,1,2); plot(time, b_data.R, "r", time, b_data.G, "g", time, b_data.B, "b", "LineWidth", 2); grid on;
        title(sprintf("Reference: Black (%s)", nombres_cota(s))); ylim([0 255]); xlabel("Time (s)"); ylabel("RGB Value");
    end

    % --- 4.2 Calibración de Ventanas ---
    for i = 1:length(sensorLabels)
        currentSensor = sensorLabels{i};
        sensor_data = dataTable(strcmp(dataTable.Window, currentSensor), :);
        
        R = sensor_data.R; G = sensor_data.G; B = sensor_data.B;
        
        if(activate_RC == 1) 
            R = (255 ./ (w_data.R - b_data.R)) .* (R - b_data.R); R(R > 255) = 255; R(R < 0) = 0;
            G = (255 ./ (w_data.G - b_data.G)) .* (G - b_data.G); G(G > 255) = 255; G(G < 0) = 0;
            B = (255 ./ (w_data.B - b_data.B)) .* (B - b_data.B); B(B > 255) = 255; B(B < 0) = 0;
        end
        
        % Extracción del feature
        Target_feature = R ./ 255;
        y_label_name = 'Red (R)';
        
        Signal_suavizada = smoothdata(Target_feature, 'movmean', 15);
        Processed_Signal_Cota(i,:) = Signal_suavizada; 

        % [GRÁFICA] Auditoría individual
        if renderizar && mostrar_graficas.individuales
            figure('Name', sprintf('Ventana %s - Cota %s', currentSensor, nombres_cota(s))) 
            subplot(2,1,1); plot((time/60), R, "r", (time/60), G, "g", (time/60), B, "b", "LineWidth", 1.5); 
            title(sprintf("Sensor: %s (%s) - Calibrated RGB", currentSensor, nombres_cota(s)), 'Interpreter', 'none');
            grid on; ylim([0 255]); xlabel("Time (minutes)"); ylabel("RGB Value"); 
            subplot(2,1,2); plot((time/60), Signal_suavizada, "k", "LineWidth", 2); 
            title(sprintf("Sensor: %s - Feature: %s", currentSensor, y_label_name), 'Interpreter', 'none');
            grid on; ylim([0, 1]); xlabel("Time (minutes)"); ylabel("Value (0-1)");
        end
        
        % --- 4.3 Separación de Fases (Histéresis) e Interpolación ---
        h_sub = hum_interpolada(idx_valle:idx_pico);
        c_sub = Signal_suavizada(idx_valle:idx_pico);
        [h_sub_unq, idx_sub_unq] = unique(h_sub); c_sub_unq = c_sub(idx_sub_unq);
        
        h_baj = hum_interpolada(idx_pico:idx_fin);
        c_baj = Signal_suavizada(idx_pico:idx_fin);
        [h_baj_unq, idx_baj_unq] = unique(h_baj); c_baj_unq = c_baj(idx_baj_unq);
        
        c_sub_interp = interp1(h_sub_unq, c_sub_unq, hum_comun, 'linear', 'extrap');
        c_baj_interp = interp1(h_baj_unq, c_baj_unq, hum_comun, 'linear', 'extrap');
        
        media_fases_all(s, i, :) = (c_sub_interp + c_baj_interp) / 2;
    end

    % --- 4.4 RENDERIZADO DE ANÁLISIS DE COTA ---
    if renderizar
        % Combinada: Perfil Memmert + Evolución Color
        figure('Name', sprintf('Combinada - Cota %s', nombres_cota(s))); 
        subplot(2,1,1); plot(time_memmert, target_memmert, 'r', 'LineWidth', 2);
        title(sprintf('Perfil de la Cámara Climática - Análisis %s', nombres_cota(s)));
        xlabel("Tiempo (minutos)"); ylabel('Humedad (%)'); grid on; xlim([0, duracion_ensayo_minutos]);

        subplot(2,1,2); hold on; grid on;
        for i = 1:length(sensorLabels)
            plot((time/60), Processed_Signal_Cota(i,:), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
        end
        title(sprintf("Respuesta Colorimétrica: %s", y_label_name));
        xlabel("Tiempo (minutos)"); ylabel(sprintf("%s (0-1)", y_label_name));
        legend('Interpreter', 'none', 'Location', 'southoutside', 'NumColumns', 4);
        xlim([0, duracion_ensayo_minutos]); ylim([0, 1]);

        % Figura Independiente Clonada
        figure('Name', sprintf('Leyenda Aislada - Cota %s', nombres_cota(s))); hold on; grid on;
        for i = 1:length(sensorLabels)
            plot((time/60), Processed_Signal_Cota(i,:), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
        end
        title(sprintf("Respuesta Colorimétrica (Aislada): %s", y_label_name));
        xlabel("Tiempo (minutos)"); ylabel(sprintf("%s (0-1)", y_label_name));
        legend('Interpreter', 'none', 'Location', 'southoutside', 'NumColumns', 4);
        xlim([0, duracion_ensayo_minutos]); ylim([0, 1]);

        % Color vs Entorno
        figure('Name', sprintf('Color vs Entorno - Cota %s', nombres_cota(s))); hold on; grid on;
        for i = 1:length(sensorLabels)
            plot(hum_interpolada(1:idx_fin), Processed_Signal_Cota(i, 1:idx_fin), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
        end
        title(sprintf("Respuesta del Sensor vs Humedad (%s)", nombres_cota(s)));
        xlabel("Humedad (%)"); ylabel(sprintf("%s (0-1)", y_label_name));
        legend('Interpreter', 'none', 'Location', 'eastoutside'); ylim([0, 1]);

        % Fase de Absorción (Subida) Pura
        figure('Name', sprintf('Absorción - Cota %s', nombres_cota(s))); hold on; grid on;
        for i = 1:length(sensorLabels)
            plot(hum_interpolada(idx_valle:idx_pico), Processed_Signal_Cota(i, idx_valle:idx_pico), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
        end
        title(sprintf("Respuesta - Absorción de Humedad - %s", nombres_cota(s)));
        xlabel("Humedad (%)"); ylabel(sprintf("%s (0-1)", y_label_name));
        legend('Interpreter', 'none', 'Location', 'eastoutside'); ylim([0, 1.05]);

        % Fase de Desorción (Bajada) Pura
        figure('Name', sprintf('Desorción - Cota %s', nombres_cota(s))); hold on; grid on;
        for i = 1:length(sensorLabels)
            plot(hum_interpolada(idx_pico:idx_fin), Processed_Signal_Cota(i, idx_pico:idx_fin), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
        end
        title(sprintf("Respuesta - Desorción de Humedad - %s", nombres_cota(s)));
        xlabel("Humedad (%)"); ylabel(sprintf("%s (0-1)", y_label_name));
        legend('Interpreter', 'none', 'Location', 'eastoutside'); ylim([0, 1.05]);

        % Media y Errorbar Interno (Absorción vs Desorción)
        figure('Position', [100, 200, 1200, 400], 'Name', sprintf('Histéresis Interna - Cota %s', nombres_cota(s))); 
        salto_barras = 20; 
        for i = 1:length(sensorLabels)
            c_sub = Processed_Signal_Cota(i, idx_valle:idx_pico); [h_sub_unq, idx_sub_unq] = unique(hum_interpolada(idx_valle:idx_pico)); c_sub_unq = c_sub(idx_sub_unq);
            c_baj = Processed_Signal_Cota(i, idx_pico:idx_fin); [h_baj_unq, idx_baj_unq] = unique(hum_interpolada(idx_pico:idx_fin)); c_baj_unq = c_baj(idx_baj_unq);
            c_sub_int = interp1(h_sub_unq, c_sub_unq, hum_comun, 'linear', 'extrap'); c_baj_int = interp1(h_baj_unq, c_baj_unq, hum_comun, 'linear', 'extrap');
            
            media_fases = (c_sub_int + c_baj_int) / 2;
            std_fases = std([c_sub_int; c_baj_int], 0, 1);
            
            subplot(2, 2, i); hold on;
            plot(hum_comun, media_fases, '-', 'Color', colors(i,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
            errorbar(hum_comun(1:salto_barras:end), media_fases(1:salto_barras:end), std_fases(1:salto_barras:end), 'o', 'Color', colors(i,:), 'MarkerSize', 5, 'MarkerFaceColor', 'none', 'LineWidth', 1.2, 'CapSize', 4);
            title(sprintf('Ventana: %s', sensorLabels{i}), 'Interpreter', 'none', 'FontSize', 10);
            xlabel('Humedad (%)'); ylabel('Red (R)'); xlim([H_min-2, H_max+2]); ylim([0, 1.05]); grid on;
            set(gca, 'Box', 'on', 'LineWidth', 1, 'GridAlpha', 0.2); 
        end
    end
end

%% 5. FUSIÓN GLOBAL MULTI-CAPA (SENSOR IDEAL)
% Cálculo de la Media y Varianza a través de la dimensión de Cotas (Bot, Mid, Up)
Media_Global = squeeze(mean(media_fases_all, 1)); 
Std_Global = squeeze(std(media_fases_all, 0, 1)); 

% --- [GRÁFICA] FUSIÓN GLOBAL ---
% Esta gráfica SIEMPRE se dibuja, independientemente de los flags
figure('Position', [100, 200, 1200, 400], 'Name', 'FUSIÓN GLOBAL: Sensor Ideal Exp3'); 
salto_barras_global = 20;

for i = 1:length(sensorLabels)
    subplot(2, 2, i); hold on;
    color_actual = colors(i,:); 
    
    plot(hum_comun, Media_Global(i,:), '-', 'Color', color_actual, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    errorbar(hum_comun(1:salto_barras_global:end), Media_Global(i, 1:salto_barras_global:end), Std_Global(i, 1:salto_barras_global:end), 'o','Color', color_actual,'MarkerSize', 5,'MarkerFaceColor', 'none','LineWidth', 1.2,'CapSize', 4);
    
    title(sprintf('Ventana: %s (Fusión Cota)', sensorLabels{i}), 'Interpreter', 'none', 'FontSize', 10);
    xlabel('Humedad (%)'); ylabel('Red (R)');
    xlim([H_min-2, H_max+2]); ylim([0, 1.05]); grid on; 
    set(gca, 'Box', 'on', 'LineWidth', 1, 'GridAlpha', 0.2); 
end
sgtitle('Sensor Ideal: Tolerancia a Gradiente Vertical y Memoria de Histéresis', 'FontWeight', 'bold');

%% 6. EXPORTACIÓN DEL CSV
nombres_columnas = {'Humedad'};
for i = 1:length(sensorLabels)
    nombres_columnas{end+1} = sensorLabels{i};
end

datos_exportar = [hum_comun', Media_Global'];
tabla_csv = array2table(datos_exportar, 'VariableNames', nombres_columnas);

% Serialización en la subcarpeta 'Datos/'
writetable(tabla_csv, 'Datos/Exp3_Resultados_Media_Global.csv');
disp('>> Archivo Exp3_Resultados_Media_Global.csv generado con éxito en la carpeta Datos/.');