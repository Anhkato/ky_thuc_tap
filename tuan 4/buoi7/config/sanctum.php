<?php

return [

    // Vì bạn không dùng cookie / csrf nên stateful để trống hoặc xóa
    'stateful' => array_filter(explode(',', env('SANCTUM_STATEFUL_DOMAINS', ''))),

    'guard' => ['web'], 

    'expiration' => null,

    'token_prefix' => env('SANCTUM_TOKEN_PREFIX', ''),

    // Bạn không dùng cookie nên có thể để middleware mặc định
    'middleware' => [
        'authenticate_session' => Laravel\Sanctum\Http\Middleware\AuthenticateSession::class,
        'encrypt_cookies' => App\Http\Middleware\EncryptCookies::class,
        'verify_csrf_token' => App\Http\Middleware\VerifyCsrfToken::class,
    ],
];
