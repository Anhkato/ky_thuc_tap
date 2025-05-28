<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\Product;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\Validator;

use Illuminate\Support\Facades\Auth;

class ProductController extends Controller
{
    public function index(Request $request)
    {
        $perPage = $request->query('per_page', 10); 
        return Product::orderBy('created_at', 'desc')->paginate($perPage);
    }

    public function store(Request $request)
    {
          if (Auth::user()->role !== 'admin') {
            abort(403, 'Bạn không có quyền thêm sản phẩm');
        }
        $validator = Validator::make($request->all(), [
            'name' => 'required|string|unique:products|max:255',
            'description' => 'nullable|string',
            'price' => 'required|numeric|min:0',
            'image_file' => 'nullable|image|mimes:jpeg,png,jpg,gif,svg|max:2048',
        ]);

        if ($validator->fails()) {
            return response()->json($validator->errors(), 422);
        }

        $data = $request->only(['name', 'description', 'price']);

        if ($request->hasFile('image_file')) {
            $path = $request->file('image_file')->store('products', 'public');
            $data['image_url'] = Storage::url($path);
        }
      

        $product = Product::create($data);
        return response()->json($product, 201);
    }

    public function show(Product $product)
    {
        return $product;
    }

    public function update(Request $request, Product $product)
    {
        if (Auth::user()->role !== 'admin') {
            abort(403, 'Bạn không có quyền thêm sửa phẩm');
        }
        $validator = Validator::make($request->all(), [
            'name' => 'required|string|max:255|unique:products,name,' . $product->id,
            'description' => 'nullable|string',
            'price' => 'required|numeric|min:0',
            'image_file' => 'nullable|image|mimes:jpeg,png,jpg,gif,svg|max:2048',
        ]);

        if ($validator->fails()) {
            return response()->json($validator->errors(), 422);
        }

        $data = $request->only(['name', 'description', 'price']);

        if ($request->hasFile('image_file')) {
            if ($product->image_url) {
                $oldPath = str_replace(Storage::url(''), '', $product->image_url);
                if (Storage::disk('public')->exists($oldPath)) {
                    Storage::disk('public')->delete($oldPath);
                }
            }
            $path = $request->file('image_file')->store('products', 'public');
            $data['image_url'] = Storage::url($path);
        }

        $product->update($data);
        return response()->json($product);
    }

    public function destroy(Product $product)
    {
        if (Auth::user()->role !== 'admin') {
            abort(403, 'Bạn không có quyền xóa sản phẩm');
        }
        if ($product->image_url) {
            $oldPath = str_replace(Storage::url(''), '', $product->image_url);
            if (Storage::disk('public')->exists($oldPath)) {
                Storage::disk('public')->delete($oldPath);
            }
        }
        $product->delete();
        return response()->json(null, 204);
    }
}