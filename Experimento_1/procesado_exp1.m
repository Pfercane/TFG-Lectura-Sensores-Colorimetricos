% =========================================================================
% SCRIPT DE PROCESAMIENTO, FUSIÓN Y CALIBRACIÓN COLORIMÉTRICA (Exp 1)
% -------------------------------------------------------------------------
% Descripción: Este script unifica el procesamiento de los sensores izquierdo 
% y derecho. Genera todas las gráficas analíticas individuales (Referencias, 
% Evolución RGB, Variable vs Entorno) y finaliza realizando una fusión 
% estadística (Media y Desviación Estándar) para exportar el modelo del 
% "Sensor Ideal".
%
% Autor: Pedro Gabriel Fernández Cañete
% Institución: Universidad de Granada (UGR)
% =========================================================================

close all;
clc;
clear all;

%% 1. CONFIGURACIÓN INICIAL
archivos_csv = ["Datos/datos_color_1_left.csv", "Datos/datos_color_1_right.csv"];
nombres_lado = ["Izquierdo", "Derecho"]; % Para titular las gráficas y no perdernos
activate_RC = 1; % Activar Calibración Radiométrica
ts = 5; % Tiempo de muestreo (segundos)

% --- MODULARIDAD ---
extraction_mode = 'I_weighted'; % Opciones: 'I_mean', 'I_weighted', 'HSV_'Canal''
show_individual_plots = false;   % true = genera una gráfica por cada ventana individual

%% 2. CARGA DE DATOS DE LA CÁMARA MEMMERT
memmert_filename = "Datos/test1.xls";
opts = detectImportOptions(memmert_filename, 'NumHeaderLines', 10);
opts = setvartype(opts, 'char'); 
memmert_data = readtable(memmert_filename, opts);

columna_objetivo = 4; % 4 = Temperatura (Columna D)
texto_variable = memmert_data{:, columna_objetivo};
texto_variable = strrep(texto_variable, ',', '.');
target_memmert = str2double(texto_variable);

num_filas = length(target_memmert);
duracion_ensayo_minutos = num_filas - 1; 
time_memmert = (0:duracion_ensayo_minutos)';

%% 3. PREPARACIÓN DE VARIABLES
tabla_temp = readtable(archivos_csv(1));
windowLabels = unique(tabla_temp.Window);
sensorLabels = windowLabels(~ismember(windowLabels, {'W_Ref', 'B_Ref'})); 
sensorLabels = sensorLabels(~strcmp(sensorLabels, 'Temp_71')); % Eliminar Temp_71

num_images = length(unique(tabla_temp.Image));
time = (0:1:num_images-1) .* ts;

% Matriz 3D: [NumSensor(2), NumVentanas(9), Tiempo]
Signal_Total = zeros(2, length(sensorLabels), num_images);
colors = lines(length(sensorLabels));

%% 4. PROCESAMIENTO BUCLE (IZQUIERDO Y LUEGO DERECHO)
for s = 1:2
    dataTable = readtable(archivos_csv(s));
    Processed_Signal_Lado = zeros(length(sensorLabels), num_images);
    
    % --- 4.1 Extracción de Referencias ---
    w_data = dataTable(strcmp(dataTable.Window, 'W_Ref'), :);
    R_w = w_data.R; G_w = w_data.G; B_w = w_data.B;
    
    b_data = dataTable(strcmp(dataTable.Window, 'B_Ref'), :);
    R_b = b_data.R; G_b = b_data.G; B_b = b_data.B;
    
    % [GRÁFICA 1] Referencias del lado actual
    figure('Name', sprintf('Referencias - Lado %s', nombres_lado(s)));
    subplot(2,1,1); 
    plot(time, R_w, "Color", [1 0 0], "LineWidth", 2); hold on; grid on;
    plot(time, G_w, "Color", [0 1 0], "LineWidth", 2); grid on;
    plot(time, B_w, "Color", [0 0 1], "LineWidth", 2); grid on;
    title(sprintf("Reference: White (%s)", nombres_lado(s))); ylim([0 255]); xlabel("Time (s)"); ylabel("RGB Value");

    subplot(2,1,2); 
    plot(time, R_b, "Color", [1 0 0], "LineWidth", 2); hold on; grid on;
    plot(time, G_b, "Color", [0 1 0], "LineWidth", 2); grid on;
    plot(time, B_b, "Color", [0 0 1], "LineWidth", 2); grid on;
    title(sprintf("Reference: Black (%s)", nombres_lado(s))); ylim([0 255]); xlabel("Time (s)"); ylabel("RGB Value");

    % --- 4.2 Calibración de cada Ventana ---
    for i = 1:length(sensorLabels)
        currentSensor = sensorLabels{i};
        sensor_data = dataTable(strcmp(dataTable.Window, currentSensor), :);
        
        R = sensor_data.R; G = sensor_data.G; B = sensor_data.B;
        
        if(activate_RC == 1) 
            R = (255 ./ (R_w - R_b)) .* (R - R_b); R(R > 255) = 255; R(R < 0) = 0;
            G = (255 ./ (G_w - G_b)) .* (G - G_b); G(G > 255) = 255; G(G < 0) = 0;
            B = (255 ./ (B_w - B_b)) .* (B - B_b); B(B > 255) = 255; B(B < 0) = 0;
        end
        
        if strcmp(extraction_mode, 'I_weighted')
            Target_feature = (0.21 .* R + 0.72 .* G + 0.07 .* B) ./ 255;
            y_label_name = 'Weighted Luminosity (Munsell)';
        else
            Target_feature = (R + G + B) ./ (3 * 255);
            y_label_name = 'Mean Intensity';
        end
        
        Signal_suavizada = smoothdata(Target_feature, 'movmean', 15);
        Processed_Signal_Lado(i,:) = Signal_suavizada; 
        Signal_Total(s, i, :) = Signal_suavizada; % Guardamos en la matriz 3D global

        % [GRÁFICA 2] Ventanas individuales
        if show_individual_plots
            figure('Name', sprintf('Ventana %s - Lado %s', currentSensor, nombres_lado(s))) 
            subplot(2,1,1); 
            plot((time/60), R, "r", (time/60), G, "g", (time/60), B, "b", "LineWidth", 1.5); 
            title(sprintf("Sensor: %s (%s) - Calibrated RGB", currentSensor, nombres_lado(s)), 'Interpreter', 'none');
            grid on; ylim([0 255]); xlabel("Time (minutes)"); ylabel("RGB Value"); 

            subplot(2,1,2);
            plot((time/60), Signal_suavizada, "k", "LineWidth", 2); 
            title(sprintf("Sensor: %s - Feature: %s", currentSensor, y_label_name), 'Interpreter', 'none');
            grid on; ylim([0, 1]); xlabel("Time (minutes)"); ylabel("Value (0-1)");
        end
    end

    % --- 4.3 [GRÁFICA 3] Combinada: Perfil Memmert + Evolución Color ---
    figure('Name', sprintf('Combinada - Lado %s', nombres_lado(s))); 
    subplot(2,1,1);
    plot(time_memmert, target_memmert, 'r', 'LineWidth', 2);
    title(sprintf('Perfil de la Cámara Climática - Análisis %s', nombres_lado(s)));
    xlabel("Tiempo (minutos)"); ylabel('Temperatura (ºC)'); grid on; xlim([0, duracion_ensayo_minutos]);

    subplot(2,1,2); hold on; grid on;
    for i = 1:length(sensorLabels)
        plot((time/60), Processed_Signal_Lado(i,:), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
    end
    title(sprintf("Respuesta Colorimétrica: %s", y_label_name));
    xlabel("Tiempo (minutos)"); ylabel(sprintf("%s (0-1)", y_label_name));
    legend('Interpreter', 'none', 'Location', 'southoutside', 'NumColumns', 4);
    xlim([0, duracion_ensayo_minutos]); ylim([0, 1]);

    % --- 4.4 [GRÁFICA 4] Figura Independiente Clonada ---
    figure('Name', sprintf('Leyenda Aislada - Lado %s', nombres_lado(s))); hold on; grid on;
    for i = 1:length(sensorLabels)
        plot((time/60), Processed_Signal_Lado(i,:), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
    end
    title(sprintf("Respuesta Colorimétrica (Aislada): %s", y_label_name));
    xlabel("Tiempo (minutos)"); ylabel(sprintf("%s (0-1)", y_label_name));
    legend('Interpreter', 'none', 'Location', 'southoutside', 'NumColumns', 4);
    xlim([0, duracion_ensayo_minutos]); ylim([0, 1]);

    % --- 4.5 [GRÁFICA 5] Variable de Color vs Variable de Entorno ---
    figure('Name', sprintf('Color vs Entorno - Lado %s', nombres_lado(s))); hold on; grid on;
    temp_interpolada = interp1(time_memmert, target_memmert, (time/60), 'linear', 'extrap');
    idx_fin = find((time/60) <= duracion_ensayo_minutos, 1, 'last');

    for i = 1:length(sensorLabels)
        plot(temp_interpolada(1:idx_fin), Processed_Signal_Lado(i, 1:idx_fin), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
    end
    title(sprintf("Respuesta del Sensor vs Temperatura (%s)", nombres_lado(s)));
    xlabel("Temperatura (ºC)"); ylabel(sprintf("%s (0-1)", y_label_name));
    legend('Interpreter', 'none', 'Location', 'eastoutside'); ylim([0, 1]);
end

%% 5. CÁLCULO DE MEDIA Y DESVIACIÓN ESTÁNDAR (FUSIÓN)
Mean_Signal = squeeze(mean(Signal_Total, 1)); 
Std_Signal = squeeze(std(Signal_Total, 0, 1)); 

temp_interpolada = interp1(time_memmert, target_memmert, (time/60), 'linear', 'extrap');
idx_fin = find((time/60) <= duracion_ensayo_minutos, 1, 'last');

temp_recortada = temp_interpolada(1:idx_fin);
Mean_recortada = Mean_Signal(:, 1:idx_fin);
Std_recortada = Std_Signal(:, 1:idx_fin);

%% 6. [GRÁFICA 6] GRÁFICAS CON BARRAS DE ERROR (SENSOR IDEAL)
figure('Position', [100, 100, 1200, 800], 'Name', 'Fusión: Barras de Error'); 
salto_barras = 10;

for i = 1:length(sensorLabels)
    subplot(2, 4, i); 
    hold on;
    color_actual = colors(i,:); 
    
    plot(temp_recortada, Mean_recortada(i,:), '-', 'Color', color_actual, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    errorbar(temp_recortada(1:salto_barras:end), Mean_recortada(i, 1:salto_barras:end), Std_recortada(i, 1:salto_barras:end), 'o','Color', color_actual,'MarkerSize', 5,'MarkerFaceColor', 'none','LineWidth', 1.2,'CapSize', 5);
    
    title(sprintf('Ventana: %s', sensorLabels{i}), 'Interpreter', 'none', 'FontSize', 10);
    xlabel('Temperatura (ºC)'); ylabel('Intensidad Media');
    xlim([min(temp_recortada)-2, max(temp_recortada)+2]); ylim([0, 1.05]);
    grid on; set(gca, 'Box', 'on', 'LineWidth', 1, 'GridAlpha', 0.2); 
end

%% 7. EXPORTACIÓN DEL CSV
nombres_columnas = {'Temperatura'};
for i = 1:length(sensorLabels)
    nombres_columnas{end+1} = sensorLabels{i};
end

datos_exportar = [temp_recortada', Mean_recortada'];
tabla_csv = array2table(datos_exportar, 'VariableNames', nombres_columnas);

% Guardamos la tabla dentro de la subcarpeta Datos
writetable(tabla_csv, 'Datos/Exp1_Resultados_Media.csv');
disp('>> Archivo Exp1_Resultados_Media.csv generado con éxito en la carpeta Datos.');