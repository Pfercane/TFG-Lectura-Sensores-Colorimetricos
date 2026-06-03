% =========================================================================
% SCRIPT DE PROCESAMIENTO, HISTÉRESIS Y FUSIÓN COLORIMÉTRICA (Exp 2)
% -------------------------------------------------------------------------
% Descripción: Este script unifica el análisis de sensores térmicos reversibles.
% Implementa compensación radiométrica y extrae el canal Rojo (RGB_R). 
% Dado el comportamiento de histéresis del material, el script separa 
% algorítmicamente la fase de calentamiento de la de enfriamiento, interpola 
% ambas curvas sobre un eje térmico común (1000 puntos) y fusiona los 
% sensores izquierdo y derecho para generar un modelo estadístico robusto.
%
% Autor: Pedro Gabriel Fernández Cañete
% Institución: Universidad de Granada (UGR)
% =========================================================================

close all;
clc;
clear all;

%% 1. CONFIGURACIÓN INICIAL Y RUTAS RELATIVAS (GITHUB)
% Las rutas apuntan al subdirectorio 'Datos/' para mantener el repositorio limpio
archivos_csv = ["Datos/datos_color_2_left.csv", "Datos/datos_color_2_right.csv"];
nombres_lado = ["Izquierdo", "Derecho"]; 
activate_RC = 1; % Flag (1/0): Calibración Radiométrica
ts = 5; % Periodo de muestreo (segundos)

% --- MODULARIDAD ---
extraction_mode = 'RGB_R'; % El Exp2 requiere aislar la degradación del canal Rojo
show_individual_plots = true; % Activar para auditar cada ventana por separado

%% 2. INGESTA DE TELEMETRÍA (CÁMARA CLIMÁTICA)
memmert_filename = "Datos/test2.xls";
opts = detectImportOptions(memmert_filename, 'NumHeaderLines', 10);
opts = setvartype(opts, 'char'); 
memmert_data = readtable(memmert_filename, opts);

columna_objetivo = 3; % 3 = Temperatura (Columna C)
texto_variable = strrep(memmert_data{:, columna_objetivo}, ',', '.');
target_memmert = str2double(texto_variable);

duracion_ensayo_minutos = length(target_memmert) - 1;
time_memmert = (0:duracion_ensayo_minutos)';

%% 3. PREPARACIÓN TOPOLÓGICA Y EJE TÉRMICO COMÚN (HISTÉRESIS)
% Lectura de metadatos desde el primer sensor
tabla_temp = readtable(archivos_csv(1));
windowLabels = unique(tabla_temp.Window);
sensorLabels = windowLabels(~ismember(windowLabels, {'W_Ref', 'B_Ref'})); 

num_images = length(unique(tabla_temp.Image));
time = (0:1:num_images-1) .* ts;

% Sincronización del tiempo fotográfico con la telemetría (Memmert)
temp_interpolada = interp1(time_memmert, target_memmert, (time/60), 'linear', 'extrap');
idx_fin = find((time/60) <= duracion_ensayo_minutos, 1, 'last');

% Detección analítica de puntos de inflexión térmica
[~, idx_pico_temp] = max(temp_interpolada(1:idx_fin));
[~, idx_valle_temp] = min(temp_interpolada(1:idx_pico_temp));

% Generación del Eje Común de Alta Resolución (1000 muestras)
T_min = max(min(temp_interpolada(idx_valle_temp:idx_pico_temp)), min(temp_interpolada(idx_pico_temp:idx_fin)));
T_max = min(max(temp_interpolada(idx_valle_temp:idx_pico_temp)), max(temp_interpolada(idx_pico_temp:idx_fin)));
temp_comun = linspace(T_min, T_max, 1000); 

% Matriz 3D para almacenar la media de histéresis: [Lado(2) x Ventanas(10) x TempComun(1000)]
media_fases_all = zeros(2, length(sensorLabels), 1000);
colors = lines(length(sensorLabels));

%% 4. MOTOR DE PROCESAMIENTO MULTI-SENSOR (Bucle Principal)
for s = 1:2
    dataTable = readtable(archivos_csv(s));
    Processed_Signal_Lado = zeros(length(sensorLabels), num_images);
    
    % --- 4.1 Aislamiento del Entorno Lumínico ---
    w_data = dataTable(strcmp(dataTable.Window, 'W_Ref'), :);
    b_data = dataTable(strcmp(dataTable.Window, 'B_Ref'), :);
    
    % [GRÁFICA 1] Referencias del lado actual
    figure('Name', sprintf('Referencias - Lado %s', nombres_lado(s)));
    subplot(2,1,1); plot(time, w_data.R, "r", time, w_data.G, "g", time, w_data.B, "b", "LineWidth", 2); grid on;
    title(sprintf("Reference: White (%s)", nombres_lado(s))); ylim([0 255]); xlabel("Time (s)"); ylabel("RGB Value");
    subplot(2,1,2); plot(time, b_data.R, "r", time, b_data.G, "g", time, b_data.B, "b", "LineWidth", 2); grid on;
    title(sprintf("Reference: Black (%s)", nombres_lado(s))); ylim([0 255]); xlabel("Time (s)"); ylabel("RGB Value");

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
        
        % Extracción estricta del canal Rojo
        Target_feature = R ./ 255;
        y_label_name = 'Red (R)';
        
        Signal_suavizada = smoothdata(Target_feature, 'movmean', 15);
        Processed_Signal_Lado(i,:) = Signal_suavizada; 

        % [GRÁFICA 2] Auditoría individual de Ventanas
        if show_individual_plots
            figure('Name', sprintf('Ventana %s - Lado %s', currentSensor, nombres_lado(s))) 
            subplot(2,1,1); plot((time/60), R, "r", (time/60), G, "g", (time/60), B, "b", "LineWidth", 1.5); 
            title(sprintf("Sensor: %s (%s) - Calibrated RGB", currentSensor, nombres_lado(s)), 'Interpreter', 'none');
            grid on; ylim([0 255]); xlabel("Time (minutes)"); ylabel("RGB Value"); 
            subplot(2,1,2); plot((time/60), Signal_suavizada, "k", "LineWidth", 2); 
            title(sprintf("Sensor: %s - Feature: %s", currentSensor, y_label_name), 'Interpreter', 'none');
            grid on; ylim([0, 1]); xlabel("Time (minutes)"); ylabel("Value (0-1)");
        end
        
        % --- 4.3 Separación de Fases (Histéresis) e Interpolación ---
        t_sub = temp_interpolada(idx_valle_temp:idx_pico_temp);
        c_sub = Signal_suavizada(idx_valle_temp:idx_pico_temp);
        [t_sub_unq, idx_sub_unq] = unique(t_sub); c_sub_unq = c_sub(idx_sub_unq);
        
        t_baj = temp_interpolada(idx_pico_temp:idx_fin);
        c_baj = Signal_suavizada(idx_pico_temp:idx_fin);
        [t_baj_unq, idx_baj_unq] = unique(t_baj); c_baj_unq = c_baj(idx_baj_unq);
        
        c_sub_interp = interp1(t_sub_unq, c_sub_unq, temp_comun, 'linear', 'extrap');
        c_baj_interp = interp1(t_baj_unq, c_baj_unq, temp_comun, 'linear', 'extrap');
        
        media_fases_all(s, i, :) = (c_sub_interp + c_baj_interp) / 2;
    end

    % --- 4.4 [GRÁFICA 3] Combinada: Perfil Memmert + Evolución Color ---
    figure('Name', sprintf('Combinada - Lado %s', nombres_lado(s))); 
    subplot(2,1,1); plot(time_memmert, target_memmert, 'r', 'LineWidth', 2);
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

    % --- 4.5 [GRÁFICA 4] Figura Independiente Clonada ---
    figure('Name', sprintf('Leyenda Aislada - Lado %s', nombres_lado(s))); hold on; grid on;
    for i = 1:length(sensorLabels)
        plot((time/60), Processed_Signal_Lado(i,:), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
    end
    title(sprintf("Respuesta Colorimétrica (Aislada): %s", y_label_name));
    xlabel("Tiempo (minutos)"); ylabel(sprintf("%s (0-1)", y_label_name));
    legend('Interpreter', 'none', 'Location', 'southoutside', 'NumColumns', 4);
    xlim([0, duracion_ensayo_minutos]); ylim([0, 1]);

    % --- 4.6 [GRÁFICA 5] Variable de Color vs Variable de Entorno ---
    figure('Name', sprintf('Color vs Entorno - Lado %s', nombres_lado(s))); hold on; grid on;
    for i = 1:length(sensorLabels)
        plot(temp_interpolada(1:idx_fin), Processed_Signal_Lado(i, 1:idx_fin), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
    end
    title(sprintf("Respuesta del Sensor vs Temperatura (%s)", nombres_lado(s)));
    xlabel("Temperatura (ºC)"); ylabel(sprintf("%s (0-1)", y_label_name));
    legend('Interpreter', 'none', 'Location', 'eastoutside'); ylim([0, 1]);

    % --- 4.7 [GRÁFICA 6] Fase de Calentamiento Pura (Datos Crudos) ---
    figure('Name', sprintf('Fase Calentamiento - Lado %s', nombres_lado(s))); hold on; grid on;
    for i = 1:length(sensorLabels)
        plot(temp_interpolada(idx_valle_temp:idx_pico_temp), Processed_Signal_Lado(i, idx_valle_temp:idx_pico_temp), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
    end
    title(sprintf("Respuesta - Solo Calentamiento (Crudo) - %s", nombres_lado(s)));
    xlabel("Temperatura (ºC)"); ylabel(sprintf("%s (0-1)", y_label_name));
    legend('Interpreter', 'none', 'Location', 'eastoutside'); ylim([0, 1.05]);

    % --- 4.8 [GRÁFICA 7] Fase de Enfriamiento Pura (Datos Crudos) ---
    figure('Name', sprintf('Fase Enfriamiento - Lado %s', nombres_lado(s))); hold on; grid on;
    for i = 1:length(sensorLabels)
        plot(temp_interpolada(idx_pico_temp:idx_fin), Processed_Signal_Lado(i, idx_pico_temp:idx_fin), "LineWidth", 2, 'DisplayName', sensorLabels{i}, 'Color', colors(i,:));
    end
    title(sprintf("Respuesta - Solo Enfriamiento (Crudo) - %s", nombres_lado(s)));
    xlabel("Temperatura (ºC)"); ylabel(sprintf("%s (0-1)", y_label_name));
    legend('Interpreter', 'none', 'Location', 'eastoutside'); ylim([0, 1.05]);

    % --- 4.9 [GRÁFICA 8] Media y Errorbar Interno (Calentamiento vs Enfriamiento) ---
    figure('Position', [50, 100, 1600, 600], 'Name', sprintf('Histéresis Interna - Lado %s', nombres_lado(s))); 
    salto_barras = 20; 
    for i = 1:length(sensorLabels)
        % Recalculamos la desviación local para dibujar el error de histéresis
        c_sub = Processed_Signal_Lado(i, idx_valle_temp:idx_pico_temp); [t_sub_unq, idx_sub_unq] = unique(temp_interpolada(idx_valle_temp:idx_pico_temp)); c_sub_unq = c_sub(idx_sub_unq);
        c_baj = Processed_Signal_Lado(i, idx_pico_temp:idx_fin); [t_baj_unq, idx_baj_unq] = unique(temp_interpolada(idx_pico_temp:idx_fin)); c_baj_unq = c_baj(idx_baj_unq);
        c_sub_int = interp1(t_sub_unq, c_sub_unq, temp_comun, 'linear', 'extrap'); c_baj_int = interp1(t_baj_unq, c_baj_unq, temp_comun, 'linear', 'extrap');
        
        media_fases = (c_sub_int + c_baj_int) / 2;
        std_fases = std([c_sub_int; c_baj_int], 0, 1);
        
        subplot(2, 5, i); hold on;
        plot(temp_comun, media_fases, '-', 'Color', colors(i,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
        errorbar(temp_comun(1:salto_barras:end), media_fases(1:salto_barras:end), std_fases(1:salto_barras:end), 'o', 'Color', colors(i,:), 'MarkerSize', 5, 'MarkerFaceColor', 'none', 'LineWidth', 1.2, 'CapSize', 4);
        title(sprintf('Ventana: %s', sensorLabels{i}), 'Interpreter', 'none', 'FontSize', 10);
        xlabel('Temperatura (ºC)'); ylabel('Red (R)'); xlim([T_min-2, T_max+2]); ylim([0, 1.05]); grid on;
        set(gca, 'Box', 'on', 'LineWidth', 1, 'GridAlpha', 0.2); 
    end
end

%% 5. FUSIÓN GLOBAL Y SENSOR IDEAL
% Cálculo de la Media y Varianza entre el Sensor Izquierdo y Derecho
Media_Global = squeeze(mean(media_fases_all, 1)); 
Std_Global = squeeze(std(media_fases_all, 0, 1)); 

% --- [GRÁFICA 9] GRÁFICA FINAL (SENSOR IDEAL FUSIONADO) ---
figure('Position', [100, 100, 1600, 600], 'Name', 'FUSIÓN GLOBAL: Sensor Ideal Exp2'); 
salto_barras_global = 20;

for i = 1:length(sensorLabels)
    subplot(2, 5, i); hold on;
    color_actual = colors(i,:); 
    
    plot(temp_comun, Media_Global(i,:), '-', 'Color', color_actual, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    errorbar(temp_comun(1:salto_barras_global:end), Media_Global(i, 1:salto_barras_global:end), Std_Global(i, 1:salto_barras_global:end), 'o','Color', color_actual,'MarkerSize', 5,'MarkerFaceColor', 'none','LineWidth', 1.2,'CapSize', 4);
    
    title(sprintf('Ventana: %s (Ideal)', sensorLabels{i}), 'Interpreter', 'none', 'FontSize', 10);
    xlabel('Temperatura (ºC)'); ylabel('Red (R)');
    xlim([T_min-2, T_max+2]); ylim([0, 1.05]); grid on; 
    set(gca, 'Box', 'on', 'LineWidth', 1, 'GridAlpha', 0.2); 
end

%% 6. EXPORTACIÓN DEL CSV
nombres_columnas = {'Temperatura'};
for i = 1:length(sensorLabels)
    nombres_columnas{end+1} = sensorLabels{i};
end

datos_exportar = [temp_comun', Media_Global'];
tabla_csv = array2table(datos_exportar, 'VariableNames', nombres_columnas);

% Serialización en la subcarpeta 'Datos/'
writetable(tabla_csv, 'Datos/Exp2_Resultados_Media_Global.csv');
disp('>> Archivo Exp2_Resultados_Media_Global.csv generado con éxito en la carpeta Datos/.');