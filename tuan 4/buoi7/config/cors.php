<?php

return [
    'paths' => [
        'api/*',
        'login',
        'logout',
        'sanctum/csrf-cookie',
    ],
    
    'allowed_methods' => ['*'],
    'allowed_origins' => [
        env('FRONTEND_URL', 'http://localhost:5173'),
    ],
    'allowed_origins_patterns' => [],
    'allowed_headers' => [
        'Content-Type',
        'X-XSRF-TOKEN',
        'X-Requested-With',
        'Accept',
        'Authorization',
    ],
    'exposed_headers' => [],
    'max_age' => 0,
    'supports_credentials' => true,
];