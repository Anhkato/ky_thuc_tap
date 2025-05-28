<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\CandidateController;

Route::get('/', [CandidateController::class, 'create'])->name('candidates.create');
Route::post('/candidates', [CandidateController::class, 'store'])->name('candidates.store');
Route::get('/home', [CandidateController::class, 'index'])->name('candidates.index');