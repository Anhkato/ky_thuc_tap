<?php

use App\Http\Controllers\ProfileController;
use Illuminate\Support\Facades\Route;
use App\Models\Post;

Route::get('/', function () {
    return view('welcome');
});

Route::get('/testcate', function () {
    $posts = Post::all();
    return view('testcate', compact('posts'));
})->middleware('auth');

// đây là phần test
Route::get('/posts/{post}/edit', function (Post $post) {
    
    if (Gate::denies('edit-post', $post)) {
        abort(403, 'Bạn không có quyền sửa bài viết này.');
    }
    return "Bạn được phép sửa bài viết: " . $post->title;
})->middleware('auth')->name('posts.edit');


Route::get('/dashboard', function () {
    return view('dashboard');
})->middleware(['auth', 'verified'])->name('dashboard');

Route::middleware('auth')->group(function () {
    Route::get('/profile', [ProfileController::class, 'edit'])->name('profile.edit');
    Route::patch('/profile', [ProfileController::class, 'update'])->name('profile.update');
    Route::delete('/profile', [ProfileController::class, 'destroy'])->name('profile.destroy');
});

require __DIR__.'/auth.php';
