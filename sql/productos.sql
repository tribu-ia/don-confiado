CREATE TABLE productos (
    id SERIAL PRIMARY KEY,
    sku VARCHAR(50) NOT NULL,
    nombre VARCHAR(200) NOT NULL,
    precio_venta NUMERIC(15, 2) NOT NULL,
    cantidad INTEGER NOT NULL DEFAULT 0,
    proveedor_id INTEGER REFERENCES terceros(id),
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT productos_sku_key UNIQUE (sku),
    CONSTRAINT productos_cantidad_check CHECK (cantidad >= 0),
    CONSTRAINT productos_precio_venta_check CHECK (precio_venta >= 0)
);


INSERT INTO productos (sku, nombre, precio_venta, cantidad, proveedor_id)
VALUES
('BVB001', 'Cerveza Club Colombia Dorada 330ml', 4500.00, 120, (SELECT id FROM terceros WHERE razon_social ILIKE '%BAVARIA%' LIMIT 1)),
('PST001', 'Gaseosa Postobón Manzana 1.5L', 3800.00, 200, (SELECT id FROM terceros WHERE razon_social ILIKE '%POSTOBON%' LIMIT 1)),
('NTR001', 'Galletas Festival Chocolate 12 und', 5200.00, 150, (SELECT id FROM terceros WHERE razon_social ILIKE '%NUTRESA%' LIMIT 1)),
('CRV001', 'Cuaderno Norma Cuadriculado 100 hojas', 8900.00, 80, (SELECT id FROM terceros WHERE razon_social ILIKE '%CARVAJAL%' LIMIT 1)),
('ARG001', 'Cemento Argos Gris 50kg', 33000.00, 60, (SELECT id FROM terceros WHERE razon_social ILIKE '%ARGOS%' LIMIT 1)),
('ECO001', 'Lubricante Industrial Ecopetrol 1L', 25000.00, 40, (SELECT id FROM terceros WHERE razon_social ILIKE '%ECOPETROL%' LIMIT 1)),
('TER001', 'Gasolina Extra Galón Terpel', 13500.00, 500, (SELECT id FROM terceros WHERE razon_social ILIKE '%TERPEL%' LIMIT 1)),
('COL001', 'Plan Corporativo Salud Colsubsidio', 78000.00, 15, (SELECT id FROM terceros WHERE razon_social ILIKE '%COLSUBSIDIO%' LIMIT 1)),
('DAV001', 'Tarjeta Prepago Davivienda', 50000.00, 100, (SELECT id FROM terceros WHERE razon_social ILIKE '%DAVIVIENDA%' LIMIT 1)),
('CCF001', 'Coca-Cola 1.5L', 4200.00, 180, (SELECT id FROM terceros WHERE razon_social ILIKE '%COCA%' LIMIT 1));
