<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Auth\LoginController;

Route::post('/login', [LoginController::class, 'login'])
    ->middleware('guest:web')
    ->name('login');

Route::post('/logout', [LoginController::class, 'logout'])
    ->middleware('auth:web')
    ->name('logout');