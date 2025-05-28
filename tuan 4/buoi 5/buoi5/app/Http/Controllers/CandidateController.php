<?php

namespace App\Http\Controllers;
use App\Http\Requests\StoreCandidateRequest;
use App\Models\Candidate;
use Illuminate\Support\Facades\Storage;
use App\Mail\CandidateMail;
use Illuminate\Support\Facades\Mail;

class CandidateController extends Controller
{
   public function create()
   {
       return view('candidates.create');
   }
   public function store(StoreCandidateRequest $request)
   {
       $data = $request->validated();

       if ($request->hasFile('avatar')) {
           $data['avatar_path'] = Storage::disk('public')->putFile('candidates', $request->file('avatar'));
       }

       if ($request->hasFile('cv')) {
           $data['cv_path'] = Storage::disk('public')->putFile('candidates', $request->file('cv'));
       }
       Mail::to($data['email'])->send(new CandidateMail());

       Candidate::create($data);

       return redirect()->route('candidates.create')->with('success', 'Hồ sơ tạo thành công!');
   }
   public function index()
   {
       $datas = \App\Models\Candidate::latest()->paginate(10);
       return view('candidates.index', compact('datas'));
   }
}
