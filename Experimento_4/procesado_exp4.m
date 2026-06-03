% =========================================================================
% SCRIPT DE PROCESAMIENTO DE SEÑAL Y CALIBRACIÓN COLORIMÉTRICA
% Experimento 4: Sensor TimeStrip
% -------------------------------------------------------------------------
% Descripción: Este script ingiere datos brutos extraídos por visión artificial, 
% resuelve inconsistencias temporales (datos faltantes), aplica calibración 
% radiométrica para eliminar el ruido de iluminación y genera un modelo de 
% "Sensor Ideal" promediado para su posterior uso en Machine Learning.
%
% Autor: Pedro Gabriel Fernández Cañete
% Institución: Universidad de Granada (UGR)
% =========================================================================

close all;
clc;
clear all;

%% 1. CONFIGURACIÓN INICIAL Y PARÁMETROS
% Rutas relativas apuntando al subdirectorio 'Datos/'
csv_filename = "Datos/datos_color_4.csv";
activate_RC = 1; % Flag (1/0): Activa la Calibración Radiométrica (Estandarización de luz)

% --- MODULARIDAD ---
extraction_mode = 'R_norm'; % Canal de interés (R_norm aísla el avance del tinte rojo)
show_individual_plots = true; % Flag (true/false): Visualización de gráficas de depuración por sensor

%% 2. INGESTA Y LIMPIEZA DE DATOS
opts = detectImportOptions(csv_filename);
opts = setvartype(opts, 'Sensor', 'string'); 
dataTable = readtable(csv_filename, opts);

sensorLabels = unique(dataTable.Sensor);
sensorLabels = sensorLabels(~ismember(sensorLabels, ["W_Ref", "B_Ref"])); 

% Filtrado de sensores defectuosos (hardware)
sensorLabels = sensorLabels(sensorLabels ~= "6"); 

% Reconstrucción del eje temporal continuo en días (Eje X)
fechas_unicas = unique(dataTable.Fecha);
fechas_str = num2str(fechas_unicas);
fechas_datetime = datetime(fechas_str, 'InputFormat', 'yyyyMMdd');
time_days = days(fechas_datetime - fechas_datetime(1));
num_images = length(time_days);

%% 3. EXTRACCIÓN Y FUSIÓN DE PATRONES DE CALIBRACIÓN
w_data = dataTable(dataTable.Sensor == "W_Ref", :);
[G_idx_w, fechas_w] = findgroups(w_data.Fecha);
R_w_global = splitapply(@mean, w_data.R, G_idx_w);
G_w_global = splitapply(@mean, w_data.G, G_idx_w);
B_w_global = splitapply(@mean, w_data.B, G_idx_w);

b_data = dataTable(dataTable.Sensor == "B_Ref", :);
[G_idx_b, fechas_b] = findgroups(b_data.Fecha);
R_b_global = splitapply(@mean, b_data.R, G_idx_b);
G_b_global = splitapply(@mean, b_data.G, G_idx_b);
B_b_global = splitapply(@mean, b_data.B, G_idx_b);

%% 4. GRÁFICAS DE ESTABILIDAD DE LAS REFERENCIAS
fechas_w_str = num2str(fechas_w);
fechas_w_dt = datetime(fechas_w_str, 'InputFormat', 'yyyyMMdd');
time_days_w = days(fechas_w_dt - fechas_datetime(1));

fechas_b_str = num2str(fechas_b);
fechas_b_dt = datetime(fechas_b_str, 'InputFormat', 'yyyyMMdd');
time_days_b = days(fechas_b_dt - fechas_datetime(1));

figure()
subplot(2,1,1); 
plot(time_days_w, R_w_global, "Color", [1 0 0], "LineWidth", 2); hold on; grid on;
plot(time_days_w, G_w_global, "Color", [0 1 0], "LineWidth", 2); grid on;
plot(time_days_w, B_w_global, "Color", [0 0 1], "LineWidth", 2); grid on;
title("Reference: White (Daily Average)");
ylim([0 255]);
xlabel("Time (Days)"); ylabel("RGB Value");

subplot(2,1,2); 
plot(time_days_b, R_b_global, "Color", [1 0 0], "LineWidth", 2); hold on; grid on;
plot(time_days_b, G_b_global, "Color", [0 1 0], "LineWidth", 2); grid on;
plot(time_days_b, B_b_global, "Color", [0 0 1], "LineWidth", 2); grid on;
title("Reference: Black (Daily Average)");
ylim([0 255]);
xlabel("Time (Days)"); ylabel("RGB Value");

%% 5. PROCESAMIENTO DIGITAL DE SEÑAL POR SENSOR
Processed_Signal = NaN(length(sensorLabels), num_images);

for i = 1:length(sensorLabels)
    currentSensor = sensorLabels(i);
    sensor_data = dataTable(dataTable.Sensor == currentSensor, :);
    
    R = sensor_data.R; G = sensor_data.G; B = sensor_data.B;

    [~, loc_w] = ismember(sensor_data.Fecha, fechas_w);
    R_w = R_w_global(loc_w); G_w = G_w_global(loc_w); B_w = B_w_global(loc_w);
    
    [~, loc_b] = ismember(sensor_data.Fecha, fechas_b);
    R_b = R_b_global(loc_b); G_b = G_b_global(loc_b); B_b = B_b_global(loc_b);

    if(activate_RC == 1) 
        R = (255 ./ (R_w - R_b)) .* (R - R_b); R(R > 255) = 255; R(R < 0) = 0;
        G = (255 ./ (G_w - G_b)) .* (G - G_b); G(G > 255) = 255; G(G < 0) = 0;
        B = (255 ./ (B_w - B_b)) .* (B - B_b); B(B > 255) = 255; B(B < 0) = 0;
    end

    switch extraction_mode
        case 'R_norm'
            Target_feature = R ./ 255; 
            y_label_name = 'Normalized Red Channel';
        otherwise
            error('Modo de extracción no reconocido.');
    end
    
    Signal_suavizada = smoothdata(Target_feature, 'movmean', 3); 
    
    [~, loc_time] = ismember(sensor_data.Fecha, fechas_unicas);
    Processed_Signal(i, loc_time) = Signal_suavizada; 

    if show_individual_plots
        figure() 
        time_sensor_days = time_days(loc_time);
        
        subplot(2,1,1); 
        plot(time_sensor_days, R, "r", time_sensor_days, G, "g", time_sensor_days, B, "b", "LineWidth", 1.5); 
        title(sprintf("Sensor: %s - Calibrated RGB", currentSensor), 'Interpreter', 'none');
        grid on; ylim([0 255]); xlabel("Time (Days)"); ylabel("RGB Value"); 

        subplot(2,1,2);
        plot(time_sensor_days, Signal_suavizada, "k", "LineWidth", 2); 
        title(sprintf("Sensor: %s - Feature: %s", currentSensor, y_label_name), 'Interpreter', 'none');
        grid on; ylim([0, 1]); xlabel("Time (Days)"); ylabel("Value (0-1)");
    end
end

Processed_Signal = fillmissing(Processed_Signal, 'linear', 2);

%% 6. VISUALIZACIÓN COMBINADA MULTI-SENSOR
figure(); hold on; grid on;
colors = lines(length(sensorLabels));

for i = 1:length(sensorLabels)
    plot(time_days, Processed_Signal(i,:), "LineWidth", 2, 'DisplayName', ['Sensor ' sensorLabels{i}], 'Color', colors(i,:));
end

title(sprintf("Respuesta Colorimétrica TimeStrip: %s", y_label_name));
xlabel("Tiempo (Días)"); ylabel(sprintf("%s (0-1)", y_label_name));
legend('Interpreter', 'none', 'Location', 'eastoutside');
ylim([0, 1]); xlim([0, max(time_days)]);

%% 7. FUSIÓN DE SENSORES Y ESTADÍSTICA (SENSOR IDEAL)
Mean_Signal = mean(Processed_Signal, 1, 'omitnan'); 
Std_Signal = std(Processed_Signal, 0, 1, 'omitnan');

%% 8. VISUALIZACIÓN DEL MODELO DE INCERTIDUMBRE
figure('Position', [200, 200, 900, 600]); hold on; 
color_linea = [0.8500 0.3250 0.0980]; 

plot(time_days, Mean_Signal, '-', 'Color', color_linea, 'LineWidth', 2, 'HandleVisibility', 'off');
errorbar(time_days, Mean_Signal, Std_Signal, 'o', 'Color', color_linea, 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'LineWidth', 1.5, 'CapSize', 5);

title(sprintf('Evolución Media de los 5 Sensores (Barras de Error)\nCaracterística: %s', y_label_name), 'Interpreter', 'none', 'FontSize', 12);
xlabel('Tiempo (Días)', 'FontSize', 11); ylabel(sprintf('%s (0-1)', y_label_name), 'FontSize', 11);
xlim([0, max(time_days) + 2]); ylim([0, 1.05]);
grid on; set(gca, 'Box', 'on', 'LineWidth', 1, 'GridAlpha', 0.2); 

%% 9. GENERACIÓN DEL DATASET PARA MACHINE LEARNING
nombres_columnas = {'Tiempo_Dias', 'Media_Color'};
datos_exportar = [time_days(:), Mean_Signal(:)]; % Serialización estricta en columnas

tabla_csv = array2table(datos_exportar, 'VariableNames', nombres_columnas);

% Serialización en la subcarpeta 'Datos/'
writetable(tabla_csv, 'Datos/Exp4_Resultados_Media.csv');
disp('>> Archivo Exp4_Resultados_Media.csv generado con éxito en Datos/.');